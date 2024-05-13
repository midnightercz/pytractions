from .base import Traction, STMD
from .tractor import Tractor, MultiTractor

import inspect


def traction_entry_points(module):
    """Discover traction entry points in given module."""
    for k in dir(module):
        v = getattr(module, k)
        if not inspect.isclass(v):
            continue
        if v in (Traction, Tractor, STMD, MultiTractor):
            continue

        if issubclass(v, Traction):
            yield f"{k} = {v.__module__}:{k}"
