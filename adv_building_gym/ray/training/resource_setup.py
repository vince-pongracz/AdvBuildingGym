"""Resource allocation for an already-built RLlib algorithm config.

Runs after ``select_model`` and before ``common_model_setup``. It is the *only*
place that touches resource-dependent settings, operating on the already
algorithm-specific config:

- splits the SLURM CPU/GPU budget into learner / driver / env-runner shares,
- resolves the algorithm-specific env-runner count (DreamerV3 forces in-process
  sampling, PPO caps to its train-batch size),
- applies ``config.learners`` and ``config.env_runners`` resource settings,
- validates the allocation against the SLURM constraints.

The resolved ``config.num_env_runners`` is later read back by ``register_callbacks``
(in ``common_model_setup``) to floor the schedule swap window. Keeping this separate
lets ``common_model_setup`` stay resource-independent.
"""

import logging

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.training.training_param_config import TrainingParamConfig
from adv_building_gym._common.resource_check_util import (
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
    """Apply algorithm-specific constraints to the resource-derived env-runner count.

    ``resource_env_runners`` is the algorithm-independent budget (all CPUs left after
    the driver and learners). The final count depends on the algorithm:

    - **DreamerV3** (Ray <= 2.52.x): ``training_step`` reads the env spaces off the
      driver-local EnvRunner (``self.env_runner.env.single_action_space``). With any
      *remote* runner the local runner has no env (``env_runner.env is None``) →
      ``AttributeError``. Force in-process sampling (0 remote runners) so the local
      runner owns the env. Fixed upstream in Ray 2.53.0 (PR #58495, reads
      ``self.spaces`` instead). Link: https://github.com/ray-project/ray/issues/56749
    - **PPO** (on-policy): RLlib validates ``total_train_batch_size ≈
      num_env_runners * rollout_fragment_length`` (within 10%). With
      ``rollout_fragment_length = episode_length`` we need
      ``num_env_runners ≈ train_batch_size_per_learner * num_learners / episode_length``.
      Cap the runner count to that target so episodes are not over-collected each
      iteration (surplus CPUs go unused).
    - All other algorithms use the full budget.
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
        train_batch = getattr(config, "train_batch_size_per_learner", None)
        if train_batch:
            total_batch = train_batch * num_learners
            target_env_runners = max(1, total_batch // episode_length)
            if resource_env_runners > target_env_runners:
                logger.warning(
                    "Reducing num_env_runners %d -> %d to match PPO total_train_batch_size=%d "
                    "(per_learner=%d x num_learners=%d) at rollout_fragment_length=%d. "
                    "Surplus CPUs will be left idle.",
                    resource_env_runners, target_env_runners, total_batch,
                    train_batch, num_learners, episode_length,
                )
                return target_env_runners

    return resource_env_runners


def resource_setup(
    config: AlgorithmConfig,
    slurm_resources: SlurmResources,
    training_config: TrainingParamConfig,
    env_config: EnvConfig,
) -> AlgorithmConfig:
    """Apply resource-dependent settings to an already-built algorithm config.

    Runs after ``select_model`` (algorithm config) and before ``common_model_setup``.
    Sets the learner / env-runner resources and the algorithm-specific env-runner
    count, then validates the allocation against the SLURM constraints. Callbacks are
    registered later by ``common_model_setup``, which reads back the resolved
    ``config.num_env_runners`` set here.

    Args:
        config: Algorithm config from ``select_model``.
        slurm_resources: SLURM-allocated CPU/GPU resources.
        training_config: Hyperparameters (``local_learner`` drives the CPU split).
        env_config: Provides ``EPISODE_LENGTH`` for the PPO env-runner cap.

    Returns:
        The same config, mutated in place.
    """
    allocation = compute_resource_allocation(slurm_resources, training_config.local_learner)

    # num_env_runners is algorithm-specific (DreamerV3 forces 0, PPO caps to batch size);
    # everything else in the allocation is algorithm-independent. Overwrite the budget
    # count with the resolved one so the struct holds the final allocation.
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
    # Sampling (querying the env / policy) — no GPU needed. rollout_fragment_length and
    # connectors are set in common_model_setup; here we only set the resource counts.
    config.env_runners(
        num_env_runners=allocation.num_env_runners,
        num_cpus_per_env_runner=allocation.num_cpus_per_env_runner,
    )

    validate_resource_allocation(allocation, slurm_resources, config.to_dict())

    return config
