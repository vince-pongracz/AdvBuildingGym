

import json
import inspect
from abc import ABCMeta
from pathlib import Path

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # numpy scalars (np.int32, np.float64, etc.)
        try:
            import numpy as np
            if isinstance(o, np.generic):
                return o.item()  # -> Python scalar
            if isinstance(o, np.ndarray):
                return o.tolist()  # -> Python list
        except Exception:
            pass

        # classes / ABCMeta (e.g., abstract base classes, types)
        if isinstance(o, ABCMeta) or inspect.isclass(o) or inspect.isroutine(o):
            mod = getattr(o, "__module__", "builtins")
            qn = getattr(o, "__qualname__", getattr(o, "__name__", str(o)))
            return f"{mod}.{qn}"

        # common extras
        if isinstance(o, Path):
            return str(o)

        # Ray objects (RLModuleSpec, RLModule, etc.)
        try:
            from ray.rllib.core.rl_module import RLModuleSpec
            if isinstance(o, RLModuleSpec):
                return {"__ray_rlmodulespec__": True, "class_name": str(o.__class__.__name__)}
        except Exception:
            pass

        # SingleAgentEpisode (RLlib) - provide meaningful serialisation
        try:
            from ray.rllib.env.single_agent_episode import SingleAgentEpisode
            if isinstance(o, SingleAgentEpisode):
                try:
                    ep_len = len(o)
                except Exception:
                    ep_len = None
                try:
                    rewards = list(o.get_rewards()) if hasattr(o, "get_rewards") else None
                except Exception:
                    rewards = None
                try:
                    custom = dict(o.custom_data) if hasattr(o, "custom_data") else {}
                except Exception:
                    custom = {}
                data = {
                    "id": getattr(o, "id_", None),
                    "env_id": getattr(o, "env_id", None),
                    "length": ep_len,
                    "total_reward": sum(rewards) if rewards is not None else None,
                    "rewards": rewards,
                    "custom_data": custom,
                }
                # Attempt to include last observations and infos if present
                try:
                    last_obs_map = getattr(o, "agent_to_last_raw_obs", None) or getattr(o, "agent_to_last_obs", None)
                    if last_obs_map:
                        data["last_obs"] = {str(k): (v.tolist() if hasattr(v, "tolist") else v) for k, v in last_obs_map.items()}
                except Exception:
                    pass
                try:
                    last_info_map = getattr(o, "agent_to_last_info", None) or getattr(o, "agent_to_last_raw_infos", None)
                    if last_info_map:
                        data["last_info"] = {str(k): v for k, v in last_info_map.items()}
                except Exception:
                    pass

                return data
        except Exception:
            pass

        # Generic Ray objects - return their string representation
        if hasattr(o, "__module__") and "ray" in str(o.__module__):
            return f"<Ray object: {o.__class__.__name__}>"
        
        try:
            return super().default(o)
        except Exception:
            return {}

