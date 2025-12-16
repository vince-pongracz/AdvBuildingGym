

import os
import sys
import time
import torch
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Dict as SDict, Box
from gymnasium import Space

import numpy as np

import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# TODO refactor this concept

def dict_space_to_space(d_space: OrderedDict[str, Space]) -> Box:
    actions_len = len(d_space)
    ret = Box(low=-1, high=1, shape=(actions_len,), dtype=np.float32)
    return ret

def dict_to_vec(actions: OrderedDict) -> np.ndarray:
    return np.array(actions.values())

def vec_to_dict(actions: np.ndarray, keys) -> OrderedDict:
    actions_dict = OrderedDict()
    i: int = 0
    
    for k in keys:
        actions[k] = actions[i]
        i += 1

    return actions_dict