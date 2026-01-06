"""Resource validation utilities for Ray/RLlib training on SLURM clusters."""

import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ResourceAllocation:
    """Container for computed resource allocation."""
    total_cpu_usage: int
    unused_cpus: int
    driver_cpus: int
    num_learners: int
    cpus_per_learner: int
    learner_total_cpus: int
    actual_env_runners: int
    cpus_per_env_runner: int
    slurm_cpus: int
    slurm_gpus: int


def validate_resource_allocation(
    allocation: ResourceAllocation,
    param_space: Dict[str, Any],
) -> None:
    """
    Validate that computed resource allocation is within SLURM constraints.

    Args:
        allocation: ResourceAllocation dataclass with computed values.
        param_space: RLlib param_space dict to check GPU config.

    Raises:
        ValueError: If allocation exceeds SLURM constraints.
    """
    # Validate CPU allocation doesn't exceed SLURM limit
    if allocation.total_cpu_usage > allocation.slurm_cpus:
        raise ValueError(
            f"Resource allocation exceeds SLURM constraint: requested {allocation.total_cpu_usage} CPUs "
            f"but only {allocation.slurm_cpus} allocated. Reduce num_learners ({allocation.num_learners}) or "
            f"cpus_per_learner ({allocation.cpus_per_learner}) or "
            f"cpus_per_env_runner ({allocation.cpus_per_env_runner})."
        )

    # Validate there's room for at least one env_runner
    fixed_reserved = allocation.driver_cpus + allocation.learner_total_cpus
    if fixed_reserved >= allocation.slurm_cpus:
        raise ValueError(
            f"Insufficient CPUs for env_runners: driver ({allocation.driver_cpus}) + "
            f"learners ({allocation.learner_total_cpus}) = {fixed_reserved} CPUs, "
            f"but only {allocation.slurm_cpus} allocated by SLURM. "
            f"Request more CPUs in your SLURM script (--cpus-per-task)."
        )

    # Warn about unused CPUs (suboptimal allocation)
    if allocation.unused_cpus > 0:
        logger.warning(
            "Suboptimal CPU usage: %d of %d CPUs unused. Consider adjusting cpus_per_env_runner "
            "or requesting fewer CPUs in SLURM to avoid wasting resources.",
            allocation.unused_cpus, allocation.slurm_cpus
        )

    # Validate GPU allocation
    num_gpus_per_learner = int(param_space.get("num_gpus_per_learner", 0))
    total_gpu_request = allocation.num_learners * num_gpus_per_learner
    if total_gpu_request > allocation.slurm_gpus:
        raise ValueError(
            f"GPU allocation exceeds SLURM constraint: learners request {total_gpu_request} GPUs "
            f"({allocation.num_learners} learners × {num_gpus_per_learner} GPUs each) "
            f"but only {allocation.slurm_gpus} available."
        )

    # Warn if GPU available but not used
    if allocation.slurm_gpus > 0 and total_gpu_request == 0:
        logger.warning(
            "GPU available (%d) but no learner configured to use it. "
            "Set num_gpus_per_learner > 0 in config to utilize GPU.",
            allocation.slurm_gpus
        )

    logger.info("Resource validation passed (allocation within SLURM constraints)")
