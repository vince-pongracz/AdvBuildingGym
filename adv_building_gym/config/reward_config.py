"""Reward configuration for AdvBuildingGym.

Data holder for the active reward function instances.  Reward creation
and scheduling is handled by ``RewardScheduleManager`` — this class only
stores the resulting list so that other parts of the codebase (eval,
serialization) can access it via ``EnvConfig.reward_config.rewards``.

Serialization is handled by ``RewardConfigSerializer`` (separation of concerns).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from adv_building_gym.rewards import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Data holder for reward function composition.

    Stores the active reward function instances.  Reward creation is
    delegated to ``RewardScheduleManager`` which reads definitions from
    YAML.  Assign the result to ``self.rewards`` before use.

    Serialization: use ``RewardConfigSerializer.save()`` / ``.load()``.
    """

    rewards: Optional[List[RewardFunction]] = None
