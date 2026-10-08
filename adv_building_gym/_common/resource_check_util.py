"""Resource validation utilities for Ray/RLlib training on SLURM clusters."""

import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SlurmResources:
    """SLURM-allocated hardware resources detected from environment variables."""
    num_cpus: int
    num_gpus: int


@dataclass
class ResourceAllocation:
    """CPU/GPU split from the SLURM allocation.

    ``num_env_runners`` starts as the budget (CPUs left after driver + learners) and may be
    lowered to an algorithm-specific count before validation. Aggregate totals are derived in
    ``validate_resource_allocation``, not stored here.
    """
    num_learners: int
    num_gpus_per_learner: int
    num_cpus_per_learner: int
    learner_total_cpus: int
    num_env_runners: int
    num_cpus_per_env_runner: int
    driver_cpus: int


def compute_resource_allocation(
    slurm_resources: SlurmResources,
    local_learner: bool,
) -> ResourceAllocation:
    """Split the SLURM allocation into learner / driver / env-runner shares (algorithm-agnostic).

    ``num_env_runners`` is the budget (all remaining CPUs); ``resource_setup`` lowers it later.
    - local_learner=True: num_learners=0, Learner runs in the driver (no separate CPU).
    - local_learner=False: one remote Learner per GPU (1 GPU + 1 CPU each).
    - Env runners: all CPUs left after learners and driver.
    """
    num_cpus_per_env_runner = 1
    driver_cpus = 1

    if local_learner:
        num_learners = 0
        num_gpus_per_learner = (slurm_resources.num_gpus or 1) if slurm_resources.num_gpus > 0 else 0
        num_cpus_per_learner = 0
        learner_total_cpus = 0
    else:
        num_learners = max(1, slurm_resources.num_gpus)
        num_gpus_per_learner = 1 if slurm_resources.num_gpus > 0 else 0
        num_cpus_per_learner = 1
        learner_total_cpus = num_learners * num_cpus_per_learner

    remaining_cpus = slurm_resources.num_cpus - learner_total_cpus - driver_cpus
    num_env_runners = max(1, remaining_cpus // num_cpus_per_env_runner)

    return ResourceAllocation(
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_learner=num_cpus_per_learner,
        learner_total_cpus=learner_total_cpus,
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        driver_cpus=driver_cpus,
    )


def validate_resource_allocation(
    allocation: ResourceAllocation,
    slurm_resources: SlurmResources,
    param_space: Dict[str, Any],
) -> None:
    """Validate the allocation against SLURM CPU/GPU limits; raises ValueError if exceeded.

    ``param_space`` is cross-checked for the applied GPU config.
    """
    # aggregate CPU usage
    total_cpu_usage = (
        allocation.driver_cpus
        + allocation.learner_total_cpus
        + allocation.num_env_runners * allocation.num_cpus_per_env_runner
    )
    unused_cpus = slurm_resources.num_cpus - total_cpu_usage

    # Validate CPU allocation doesn't exceed SLURM limit
    if total_cpu_usage > slurm_resources.num_cpus:
        raise ValueError(
            f"Resource allocation exceeds SLURM constraint: requested {total_cpu_usage} CPUs "
            f"but only {slurm_resources.num_cpus} allocated. Reduce num_learners ({allocation.num_learners}) or "
            f"num_cpus_per_learner ({allocation.num_cpus_per_learner}) or "
            f"num_cpus_per_env_runner ({allocation.num_cpus_per_env_runner})."
        )

    # Validate there's room for at least one env_runner
    fixed_reserved = allocation.driver_cpus + allocation.learner_total_cpus
    if fixed_reserved >= slurm_resources.num_cpus:
        raise ValueError(
            f"Insufficient CPUs for env_runners: driver ({allocation.driver_cpus}) + "
            f"learners ({allocation.learner_total_cpus}) = {fixed_reserved} CPUs, "
            f"but only {slurm_resources.num_cpus} allocated by SLURM. "
            f"Request more CPUs in your SLURM script (--cpus-per-task)."
        )

    # Warn about unused CPUs (suboptimal allocation)
    if unused_cpus > 0:
        logger.warning(
            "Suboptimal CPU usage: %d of %d CPUs unused. Consider adjusting num_cpus_per_env_runner "
            "or requesting fewer CPUs in SLURM to avoid wasting resources.",
            unused_cpus, slurm_resources.num_cpus
        )

    # Validate GPU allocation
    num_gpus_per_learner = int(param_space.get("num_gpus_per_learner", 0))
    total_gpu_request = allocation.num_learners * num_gpus_per_learner
    if total_gpu_request > slurm_resources.num_gpus:
        raise ValueError(
            f"GPU allocation exceeds SLURM constraint: learners request {total_gpu_request} GPUs "
            f"({allocation.num_learners} learners × {num_gpus_per_learner} GPUs each) "
            f"but only {slurm_resources.num_gpus} available."
        )

    # warn if GPU is available but unused; with num_learners=0 the driver-side learner
    # still claims it via num_gpus_per_learner, so check that, not total_gpu_request
    if slurm_resources.num_gpus > 0 and num_gpus_per_learner == 0:
        logger.warning(
            "GPU available (%d) but no learner configured to use it. "
            "Set num_gpus_per_learner > 0 in config to utilize GPU.",
            slurm_resources.num_gpus
        )

    logger.info("Resource validation passed (allocation within SLURM constraints)")
