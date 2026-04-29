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
from ray.rllib.env.utils.infinite_lookback_buffer import InfiniteLookbackBuffer
from ray.rllib.utils.numpy import flatten_inputs_to_1d_tensor
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import EpisodeType

# Cache key for per-step augmented obs persisted on the episode. Stored under
# `custom_data` (NOT `extra_model_outputs`) because validate() asserts every
# extra_model_outputs entry has len == len(observations) - 1, which our buffer
# violates transiently between env-to-module and add_env_step. custom_data is
# unvalidated and dropped on slice() — so the learner side cache-misses and
# recomputes via _augment_episode, which is fine.
HST_CACHE_KEY = "__hst_obs_cache__"

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

        # Offset 0 (current step) is always the first slice of the stack.
        # Any user-supplied 0 is dropped to avoid duplicating the current frame.
        self.tracked_keys: List[str] = list(tracked_keys)
        self._as_learner_connector = as_learner_connector
        self.offsets: List[int] = [0] + [ofs for ofs in offsets if ofs != 0]
        if any(ofs > 0 for ofs in offsets):
            raise ValueError(f"offsets must be <= 0 (0 = current step). Got: {offsets}")

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
            obs_space = env_obs_space.spaces[key]

            # Ensure type correctness: the connector relies on Box spaces.
            assert isinstance(obs_space, gym.spaces.Box), (f"tracked obs key '{key}' must be a Box, got {type(obs_space).__name__}")

            # Overwrite the original key's space with the stacked shape
            nn_obs_space[key] = gym.spaces.Box(
                low=np.broadcast_to(obs_space.low, (hst_window_len, *obs_space.shape)).astype(obs_space.dtype).copy(),
                high=np.broadcast_to(obs_space.high, (hst_window_len, *obs_space.shape)).astype(obs_space.dtype).copy(),
                shape=(hst_window_len, *obs_space.shape),
                dtype=obs_space.dtype,
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

    # TODO VP 2026.04.28. : Fallback obs rather should be null -- if no history available, do not propagate the current one...
    def _obs_at(self, sa_episode, absolute_idx: int, fallback_obs: dict, key: str):
        if absolute_idx < 0:
            return fallback_obs[key]
        try:
            obs = sa_episode.get_observations(absolute_idx)
        except (IndexError, KeyError):
            return fallback_obs[key]

        if (obs is None or
            not isinstance(obs, dict) or 
            key not in obs):
            return fallback_obs[key]
        return obs[key]

    def _add_hst_values(self, sa_episode, t: int) -> dict:
        """Extend the t-th observation in the episode with it's historic values

        Args:
            sa_episode (_type_): Episode to pull history from. Msut be raw Dict obs type (no flattening yet).
            t (int): timestep index to pull history for.

        Returns:
            dict: _description_
        """
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

    def _augment_episode(self, sa_episode) -> List[np.ndarray]:
        """Vectorised: build flat augmented obs for every t in [0, len].

        Pulls each tracked key's full trajectory once, then performs strided
        gather with NumPy fancy-indexing. Returns a list of length `len+1`
        where item ``i`` is the flat augmented obs at absolute index ``i``.
        """
        all_obs = sa_episode.get_observations()  # length len+1, list of dicts
        T_plus_1 = len(all_obs)
        self._ensure_spaces_initialized(all_obs[0])

        # Per-key buffer of shape (T+1, *obs_shape) — one stack per key, not per t.
        key_bufs: Dict[str, np.ndarray] = {
            key: np.stack([np.asarray(o[key]) for o in all_obs], axis=0)
            for key in self.tracked_keys
        }

        # Index matrix (T+1, O): row t holds [t, t+ofs1, t+ofs2, ...].
        # Negative indices fall back to t=0 (start-of-episode value), matching the
        # spirit of the original `_obs_at` boundary handling at scale.
        offs = np.asarray(self.offsets, dtype=np.int64)
        t_grid = np.arange(T_plus_1, dtype=np.int64)[:, None] + offs[None, :]
        np.clip(t_grid, 0, T_plus_1 - 1, out=t_grid)

        # Gather: (T+1, O, *obs_shape) per tracked key.
        gathered = {key: key_bufs[key][t_grid] for key in self.tracked_keys}

        # Build the flat obs per timestep. Only the final dict assembly + flatten
        # remains per-t; the expensive inner loop is gone.
        out_list: List[np.ndarray] = []
        for t in range(T_plus_1):
            out = dict(all_obs[t])
            for key in self.tracked_keys:
                out[key] = gathered[key][t]
            out_list.append(self._flatten(out))
        return out_list

    def _flatten(self, augmented_obs: dict) -> np.ndarray:
        return flatten_inputs_to_1d_tensor(inputs=augmented_obs, spaces_struct=self._flatten_struct, batch_axis=False)

    def _get_or_create_cache_buf(self, sa_episode) -> InfiniteLookbackBuffer:
        buf = sa_episode.custom_data.get(HST_CACHE_KEY)
        if buf is None:
            buf = InfiniteLookbackBuffer()
            sa_episode.custom_data[HST_CACHE_KEY] = buf
        return buf

    def _read_cached_items(self, sa_episode, T: int) -> Optional[List[np.ndarray]]:
        """Return cached augmented obs for indices 0..T-1, or None on cache miss.

        custom_data is dropped by SingleAgentEpisode.slice(), so on the learner
        side this returns None for replay-sampled episodes and the caller falls
        back to _augment_episode.
        """
        buf = sa_episode.custom_data.get(HST_CACHE_KEY)
        if buf is None or len(buf) < T:
            return None
        # buf.data may be a list (pre-numpy'ize) or ndarray (post). Both index the same.
        items = [np.asarray(buf.data[t]) for t in range(T)]
        return items

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
                T = len(sa_episode)
                if T == 0:
                    continue

                self._ensure_spaces_initialized(sa_episode.get_observations(0))

                # Try the rollout-time cache first: it holds aug(0..T-1).
                cached = self._read_cached_items(sa_episode, T)
                if cached is not None:
                    # Bootstrap aug(T) was never queried at rollout (no policy
                    # call after the terminal step) — compute the one extra now.
                    bootstrap = self._flatten(self._add_hst_values(sa_episode, T))
                    obs_full = cached + [bootstrap]   # length T+1
                else:
                    # Cache miss (e.g. episode never went through env-to-module,
                    # or hst config changed mid-run). Fall back to the vectorised
                    # path; cost is the same as the pre-cache implementation.
                    obs_full = self._augment_episode(sa_episode)   # length T+1

                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.OBS,
                    items_to_add=obs_full[:T],
                    num_items=T,
                    single_agent_episode=sa_episode,
                )
                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.NEXT_OBS,
                    items_to_add=obs_full[1:T + 1],
                    num_items=T,
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

                # Reuse a cached entry if env-to-module ran more than once for
                # the same step (rare); otherwise compute and cache.
                buf = self._get_or_create_cache_buf(sa_episode)
                if len(buf) > t_current:
                    flat = np.asarray(buf.data[t_current])
                else:
                    flat = self._flatten(self._add_hst_values(sa_episode, t_current))
                    # Persist for the learner connector. The work is already done
                    # (the policy needs `flat` regardless), so the only extra cost
                    # is the buffer append itself.
                    buf.append(flat)

                self.add_batch_item(
                    batch=batch,
                    column=Columns.OBS,
                    item_to_add=flat,
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
