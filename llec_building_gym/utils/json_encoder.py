

import json
import inspect
from abc import ABCMeta
from pathlib import Path

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # numpy scalars (np.int32, np.float64, etc.)
        try:
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()  # -> Python scalar
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # -> Python list
        except Exception:
            pass

        # classes / ABCMeta (e.g., abstract base classes, types)
        if isinstance(obj, ABCMeta) or inspect.isclass(obj) or inspect.isroutine(obj):
            mod = getattr(obj, "__module__", "builtins")
            qn = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
            return f"{mod}.{qn}"

        # common extras
        if isinstance(obj, Path):
            return str(obj)

        return super().default(obj)
