"""Per-key strided history connector for the RLlib new API stack.

For each user-listed raw observation key ``X``, the connector *replaces*
``X`` in the Dict observation with a stacked tensor of shape
``(len(offsets), *X.shape)`` containing ``X`` at each requested offset,
then flattens the augmented Dict into a single 1-D tensor for the MLP
RLModule.  Offset ``0`` (the current step) is always the first slice;
the user's configured offsets are appended after it (any duplicate
``0`` in the config is removed).

The connector is intentionally wired to run *after*
``AddObservationsFromEpisodesToBatch`` in both the env-to-module and
learner pipelines (see :func:`build_env_to_module_connectors`).  The
upstream default populates ``batch[OBS]`` with the raw Dict from the
episode; this connector then overwrites ``batch[OBS]`` with the
per-episode stacked-and-flattened tensor.  Any ``AddObservationsFromEpisodesToBatch``
that the outer ``AlgorithmConfig`` still appends at the end of the
pipeline early-outs on ``Columns.OBS in batch`` and therefore cannot
overwrite our output.

The connector deliberately does *not* mutate the episode's stored
observations (unlike RLlib's ``FlattenObservations``), because
mid-episode history lookup via ``sa_episode.get_observations(offset)``
must still return the raw Dict.

Action history: the env publishes ``<action_key>_prev`` obs entries
each step (see ``building_adv.py:step``), so they are plain obs keys
and can be listed in ``hst_tracked_keys`` alongside any other obs
signal.  The connector therefore never touches the episode's stored
actions and does not need the Dict action space.

Link: docs/hst_mgmt.md
"""

# NOTE VP 2026.04.23. : https://docs.ray.io/en/latest/rllib/learner-connector.html
# NOTE VP 2026.04.23. : https://docs.ray.io/en/latest/rllib/env-to-module-connector.html

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.numpy import flatten_inputs_to_1d_tensor
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import EpisodeType

logger = logging.getLogger(__name__)


class StridedHistoryConnector(ConnectorV2):
    """Extend selected obs keys along user-specified offsets, then flatten."""

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        tracked_keys: List[str],
        offsets: List[int],
        as_learner_connector: bool = False,
        **kwargs,
    ):
        # Param checks
        if not tracked_keys:
            raise ValueError("StridedHistoryConnector requires a non-empty tracked_keys list.")
        if not offsets:
            raise ValueError("StridedHistoryConnector requires a non-empty offsets list.")
        if any(ofs > 0 for ofs in offsets):
            raise ValueError(f"offsets must be <= 0 (0 = current step). Got: {offsets}")

        # Offset 0 (current step) is always the first slice of the stack.
        # Any user-supplied 0 is dropped to avoid duplicating the current frame.
        self.tracked_keys: List[str] = list(tracked_keys)
        self.offsets: List[int] = [0] + [ofs for ofs in offsets if ofs != 0]
        self._as_learner_connector = as_learner_connector

        # Spaces may be deferred on the learner side: RLlib calls
        # ``build_learner_connector`` with ``input_observation_space=None``
        # (see learner.py:build: TODO (sven): Figure out which space to
        # provide here). When that happens the augmented space and flatten
        # struct are inferred from the first observation we see in __call__.
        self._augmented_dict_space: Optional[gym.spaces.Dict] = None
        self._flatten_struct: Any = None

        if isinstance(input_observation_space, gym.spaces.Dict):
            self._init_spaces_from_dict(input_observation_space)
        elif input_observation_space is None:
            logger.info(
                "StridedHistoryConnector (%s): input_observation_space is None; "
                "will infer augmented space lazily from first observation.",
                "learner" if as_learner_connector else "env-to-module",
            )
        else:
            raise TypeError(
                "StridedHistoryConnector expects a Dict observation space "
                f"(or None, to defer), got {type(input_observation_space).__name__}."
            )

        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            **kwargs,
        )

    def _init_spaces_from_dict(self, env_obs_space: gym.spaces.Dict) -> None:
        """Validate tracked keys and build the augmented Dict + flatten struct."""
        obs_keys = set(env_obs_space.spaces.keys())
        unknown_keys = [key for key in self.tracked_keys if key not in obs_keys]
        if unknown_keys:
            raise KeyError(
                f"StridedHistoryConnector: tracked keys {unknown_keys} not found in "
                f"observation space ({sorted(obs_keys)}). For action history, "
                "list the '<action_key>_prev' obs entries published by the env "
                "(see building_adv.py:step)."
            )
        self._augmented_dict_space = self._build_hst_augmented_space(env_obs_space)
        self._flatten_struct = get_base_struct_from_space(self._augmented_dict_space)

    def _build_hst_augmented_space(self, env_obs_space: gym.spaces.Dict) -> gym.spaces.Dict:
        hst_window_len = len(self.offsets)
        nn_obs_space = dict(env_obs_space.spaces)

        for key in self.tracked_keys:
            box = env_obs_space.spaces[key]

            # Ensure type correctness: the connector relies on Box spaces.
            assert isinstance(box, gym.spaces.Box), (f"tracked obs key '{key}' must be a Box, got {type(box).__name__}")

            # Overwrite the original key's space with the stacked shape
            nn_obs_space[key] = gym.spaces.Box(
                low=np.broadcast_to(box.low, (hst_window_len, *box.shape)).astype(box.dtype).copy(),
                high=np.broadcast_to(box.high, (hst_window_len, *box.shape)).astype(box.dtype).copy(),
                shape=(hst_window_len, *box.shape),
                dtype=box.dtype,
            )

        return gym.spaces.Dict(nn_obs_space)

    def _infer_dict_space_from_sample(self, sample: dict) -> gym.spaces.Dict:
        """Infer a Dict space from a numpy sample observation.

        Used on the learner side, where RLlib passes
        ``input_observation_space=None``. Only shapes/dtypes are
        preserved — ``flatten_inputs_to_1d_tensor`` uses the struct
        solely for Discrete one-hotting, which does not apply here
        (all tracked entries are Box).
        """
        spaces_dict: Dict[str, gym.spaces.Space] = {}
        for key, val in sample.items():
            arr = np.asarray(val)
            # flatten_inputs_to_1d_tensor only uses shape info for Box, so
            # we standardize on float32 regardless of the observed dtype.
            spaces_dict[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
        return gym.spaces.Dict(spaces_dict)

    def _ensure_spaces_initialized(self, sample_obs: dict) -> None:
        # Already initialised
        if self._flatten_struct is not None:
            return

        assert isinstance(sample_obs, dict), (
            "StridedHistoryConnector: expected a Dict observation to infer spaces; "
            f"got {type(sample_obs).__name__}."
        )

        inferred = self._infer_dict_space_from_sample(sample_obs)
        self._init_spaces_from_dict(inferred)
        # Keep the output-space cache consistent with the freshly built struct.
        act_sp = self.input_action_space
        if act_sp is not None:
            self._observation_space = self.recompute_output_observation_space(inferred, act_sp)

    def recompute_output_observation_space(
        self,
        input_observation_space: gym.Space,
        input_action_space: gym.Space,
    ) -> gym.Space:
        # When we have no struct yet (deferred init on the learner), leave the
        # output space unset — RLlib tolerates this because the RLModule input
        # space is determined from the env-to-module side, not the learner.
        if self._flatten_struct is None:
            return None  # type: ignore[return-value]

        # Flatten an arbitrary sample to discover the output dimension.
        sample = flatten_inputs_to_1d_tensor(
            tree.map_structure(lambda s: s.sample(), self._flatten_struct),
            self._flatten_struct,
            batch_axis=False,
        )
        return gym.spaces.Box(float("-inf"), float("inf"), (len(sample),), np.float32)

    def _obs_at(self, sa_episode, absolute_idx: int, fallback_obs: dict, key: str):
        if absolute_idx < 0:
            return fallback_obs[key]
        try:
            obs = sa_episode.get_observations(absolute_idx)
        except (IndexError, KeyError):
            return fallback_obs[key]

        if obs is None or not isinstance(obs, dict) or key not in obs:
            return fallback_obs[key]
        return obs[key]

    def _augment_single(self, sa_episode, t: int) -> dict:
        current_obs = sa_episode.get_observations(t)
        assert isinstance(current_obs, dict), (
            "StridedHistoryConnector requires raw Dict observations; ensure no "
            "earlier pipeline step has flattened the observation."
        )

        out = dict(current_obs)
        for key in self.tracked_keys:
            frames = [
                np.asarray(self._obs_at(sa_episode, t + ofs, current_obs, key))
                for ofs in self.offsets
            ]
            # Replace the scalar/vector entry with the temporal stack.
            out[key] = np.stack(frames, axis=0)

        return out

    def _flatten(self, augmented_obs: dict) -> np.ndarray:
        return flatten_inputs_to_1d_tensor(inputs=augmented_obs, spaces_struct=self._flatten_struct, batch_axis=False)

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        # Option 2 wiring: upstream default connectors have already populated
        # `batch[OBS]` (and, on the SAC/DQN learner, `batch[NEXT_OBS]`) with
        # raw Dict obs. We overwrite both columns with stacked-and-flattened
        # tensors built from the episodes themselves (episodes stay pristine
        # for history lookup).
        # Link: ray/rllib/algorithms/sac/sac.py:549 — SAC inserts
        # AddNextObservationsFromEpisodesToTrainBatch right after AddObs,
        # so NEXT_OBS lands in batch as a raw Dict before this connector runs.
        
        # Remove the raw Dict obs entries added by the upstream AddObservationsFromEpisodesToBatch
        batch.pop(Columns.OBS, None)
        if self._as_learner_connector:
            batch.pop(Columns.NEXT_OBS, None)

        if self._as_learner_connector:
            for sa_episode in self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False):
                if len(sa_episode) == 0:
                    continue

                self._ensure_spaces_initialized(sa_episode.get_observations(0))
                # OBS: augmented obs at each step t ∈ [0, len-1].
                obs_items = [
                    self._flatten(self._augment_single(sa_episode, t))
                    for t in range(len(sa_episode))
                ]
                # NEXT_OBS: augmented obs at t+1 ∈ [1, len], mirroring
                # AddNextObservationsFromEpisodesToTrainBatch's slice(1, len+1).
                next_obs_items = [
                    self._flatten(self._augment_single(sa_episode, t + 1))
                    for t in range(len(sa_episode))
                ]
                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.OBS,
                    items_to_add=obs_items,
                    num_items=len(obs_items),
                    single_agent_episode=sa_episode,
                )
                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.NEXT_OBS,
                    items_to_add=next_obs_items,
                    num_items=len(next_obs_items),
                    single_agent_episode=sa_episode,
                )
        else:
            for sa_episode in self.single_agent_episode_iterator(episodes):
                # Index of the current observation: get_observations(-1) returns
                # the most recent stored obs; its absolute index is len(sa_episode)
                # (obs list has len+1 entries: reset + one per env step).
                t_current = len(sa_episode)
                current_obs = sa_episode.get_observations(t_current)
                self._ensure_spaces_initialized(current_obs)
                self.add_batch_item(
                    batch=batch,
                    column=Columns.OBS,
                    item_to_add=self._flatten(self._augment_single(sa_episode, t_current)),
                    single_agent_episode=sa_episode,
                )

        return batch


def build_env_to_module_connectors(
    training_config,
    input_observation_space: gym.Space,
    input_action_space: gym.Space,
    as_learner_connector: bool = False,
):
    """Shared factory for train, SAC-learner, and eval pipelines.

    Returns the list of ConnectorV2 pieces that collectively produce a
    flat observation for the RLModule.

    The returned order implements Option 2 (see module docstring):
    ``[AddObservationsFromEpisodesToBatch, StridedHistoryConnector]``.
    The upstream ``AddObservationsFromEpisodesToBatch`` populates
    ``batch[Columns.OBS]`` with raw Dict obs so downstream callers and
    debuggers can inspect it; ``StridedHistoryConnector`` then replaces
    the entry with the stacked-and-flattened tensor. Any further default
    ``AddObservationsFromEpisodesToBatch`` that ``AlgorithmConfig``
    appends at the end of the pipeline early-outs on
    ``Columns.OBS in batch`` and is a no-op.

    When ``hst_tracked_keys`` is empty the pipeline degrades to the
    stock ``[FlattenObservations]``.

    Missing tracked keys are dropped with a warning (e.g. when an env
    variant doesn't publish an expected ``<action_key>_prev`` entry), so
    a single YAML config can be reused across env configs with differing
    infrastructure.  If all tracked keys are absent the pipeline also
    falls back to ``[FlattenObservations]``.

    ``input_observation_space`` may be ``None`` on the learner side (RLlib
    passes None there — see ``core/learner/learner.py:build``); the
    connector defers space construction until the first observation is
    seen at call time, and ``input_action_space`` is accepted for
    ConnectorV2 base-class compatibility only.
    """
    
    # TODO VP 2026.04.23. : Maybe switch out the default connectors and set them up by hand (so set it up, not just use them implicitly via the default pipeline).
    
    tracked_keys = list(getattr(training_config, "hst_tracked_keys", []) or [])
    offsets = list(getattr(training_config, "hst_offsets", []) or [])

    if not (tracked_keys and offsets):
        return [FlattenObservations()]

    # Only validate tracked keys against the env obs space when we actually
    # have a Dict (env-to-module path).  On the learner the obs space is
    # None, so defer validation to the first __call__.
    if isinstance(input_observation_space, gym.spaces.Dict):
        env_obs_keys = set(input_observation_space.spaces.keys())
        missing_keys = [k for k in tracked_keys if k not in env_obs_keys]
        if missing_keys:
            logger.warning(
                "StridedHistoryConnector: skipping tracked keys %s (not present in "
                "observation space). History will not be stored for these keys. "
                "Observation keys available: %s",
                missing_keys, sorted(env_obs_keys),
            )
            tracked_keys = [k for k in tracked_keys if k in env_obs_keys]
        if not tracked_keys:
            logger.warning(
                "StridedHistoryConnector: no valid tracked keys remain after "
                "filtering; falling back to FlattenObservations (no history stacking)."
            )
            return [FlattenObservations()]

    # Ordering: explicit AddObservationsFromEpisodesToBatch first so
    # batch[OBS] holds the raw Dict before StridedHistoryConnector runs and
    # overwrites it.  The same connector piece is also appended by RLlib's
    # default pipeline at the end; that second copy sees OBS already populated
    # (by us) and early-outs, so no duplicate work is performed.
    # Link: https://github.com/ray-project/ray/blob/master/rllib/connectors/common/add_observations_from_episodes_to_batch.py
    return [
        AddObservationsFromEpisodesToBatch(as_learner_connector=as_learner_connector),
        StridedHistoryConnector(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            tracked_keys=tracked_keys,
            offsets=offsets,
            as_learner_connector=as_learner_connector,
        ),
    ]
