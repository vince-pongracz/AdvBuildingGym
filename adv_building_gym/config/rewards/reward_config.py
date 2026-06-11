"""Reward configuration — holds the active reward function instances.

Creation/scheduling is done by ``RewardScheduleManager``; serialisation by
``RewardConfigSerializer``. Accessed via ``EnvConfig.reward_config.rewards``.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from adv_building_gym.config.utils.loggable_config import LoggableConfig
from adv_building_gym.components.rewards import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig(LoggableConfig):
    """Data holder for reward function composition.

    Stores the active reward function instances.
    Reward creation is delegated to ``RewardScheduleManager`` which reads definitions from
    YAML.
    Assign the result to ``self.rewards`` before use.

    Serialization: use ``RewardConfigSerializer.save()`` / ``.load()``.
    """

    rewards: Optional[List[RewardFunction]] = None

    def _log_label(self) -> str:
        return "RewardConfig"

    def log_values(self) -> None:
        """Log reward function details at INFO level."""
        if self.rewards is None:
            logger.info("%s: no rewards configured", self._log_label())
            return
        lines = [
            f"  {r.name}: weight={r.weight}, class={type(r).__name__}"
            for r in self.rewards
        ]
        logger.info(
            "%s (%d rewards):\n%s",
            self._log_label(), len(self.rewards), "\n".join(lines),
        )
