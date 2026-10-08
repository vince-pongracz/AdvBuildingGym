"""Resource allocation for an already-built RLlib algorithm config.

Runs between ``select_model`` and ``common_model_setup`` — the only place touching
resource settings: splits the SLURM CPU/GPU budget into learner/driver/env-runner shares,
resolves the algorithm-specific env-runner count (DreamerV3 → 0, PPO → all CPUs up to one
runner per episode, rounding the train batch up to whole sampling rounds), applies
``config.learners`` / ``config.env_runners``, and validates against SLURM.
``common_model_setup`` reads back the resolved ``config.num_env_runners``.
"""

import logging
import math

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.training.training_param_config import TrainingParamConfig
from adv_building_gym._common.resource_check_util import (
    ResourceAllocation,
    SlurmResources,
    compute_resource_allocation,
    validate_resource_allocation,
)

logger = logging.getLogger(__name__)


def resolve_num_env_runners(
    config: AlgorithmConfig,
    resource_env_runners: int,
    episode_length: int,
    num_learners: int,
) -> int:
    """Apply algorithm-specific constraints to the budget env-runner count.

    - DreamerV3 (Ray ≤ 2.52.x): force 0 remote runners — training_step reads env spaces off
      the driver-local runner, which has no env when remote runners exist. Fixed in Ray 2.53.0.
      Link: https://github.com/ray-project/ray/issues/56749
    - PPO: full budget, capped at one runner per episode; see ``_resolve_ppo_env_runners``.
    - Others: full budget.
    """
    algo = type(config).__name__

    if algo == "DreamerV3Config":
        if resource_env_runners != 0:
            logger.warning(
                "DreamerV3 requires in-process sampling on Ray < 2.53.0; "
                "forcing num_env_runners %d -> 0.", resource_env_runners,
            )
        return 0

    if algo == "PPOConfig":
        return _resolve_ppo_env_runners(
            config, resource_env_runners, episode_length, num_learners,
        )

    return resource_env_runners

def _resolve_ppo_env_runners(
    config: AlgorithmConfig,
    resource_env_runners: int,
    episode_length: int,
    num_learners: int,
) -> int:
    """Use the full CPU budget (max one runner per episode); round the train batch up to
    whole sampling rounds — each round polls ALL runners for one complete episode, and
    RLlib rejects a train batch that is not a multiple of runners × episode_length.
    """
    # A local learner (num_learners=0) counts as one, mirroring RLlib's total_train_batch_size.
    learners = num_learners or 1
    episodes_per_iteration = max(1, config.train_batch_size_per_learner * learners // episode_length)
    num_env_runners = min(resource_env_runners, episodes_per_iteration)

    episodes_aligned = math.ceil(episodes_per_iteration / num_env_runners) * num_env_runners
    if episodes_aligned != episodes_per_iteration:
        logger.warning(
            "PPO: raising episodes/iteration %d -> %d (next multiple of %d env runners).",
            episodes_per_iteration, episodes_aligned, num_env_runners,
        )
        config.training(train_batch_size_per_learner=episodes_aligned * episode_length // learners)
    return num_env_runners


def resource_setup(
    config: AlgorithmConfig,
    slurm_resources: SlurmResources,
    training_config: TrainingParamConfig,
    env_config: EnvConfig,
) -> AlgorithmConfig:
    """Apply resource-dependent settings to an algorithm config (mutated in place).

    Sets learner/env-runner resources and the algorithm-specific env-runner count, then
    validates against SLURM. ``common_model_setup`` later reads back ``config.num_env_runners``.
    """
    allocation: ResourceAllocation = compute_resource_allocation(slurm_resources, training_config.local_learner)

    # num_env_runners is algorithm-specific; overwrite the budget with the resolved count
    allocation.num_env_runners = resolve_num_env_runners(
        config, allocation.num_env_runners, env_config.EPISODE_LENGTH, allocation.num_learners,
    )

    logger.info(
        "Resource allocation in rllib_config: learners=%d (gpus=%d, cpus=%d each), "
        "env_runners=%d (cpus=%d each), driver=%d CPU",
        allocation.num_learners, allocation.num_gpus_per_learner, allocation.num_cpus_per_learner,
        allocation.num_env_runners, allocation.num_cpus_per_env_runner, allocation.driver_cpus,
    )

    # Learning the NN / policy (gradient updates) — needs GPU.
    config.learners(
        num_learners=allocation.num_learners,
        num_gpus_per_learner=allocation.num_gpus_per_learner,
        num_cpus_per_learner=allocation.num_cpus_per_learner,
    )
    # Sampling — no GPU; fragment_length/connectors are in common_model_setup, only counts here
    config.env_runners(
        num_env_runners=allocation.num_env_runners,
        num_cpus_per_env_runner=allocation.num_cpus_per_env_runner,
    )

    validate_resource_allocation(allocation, slurm_resources, config.to_dict())

    return config
