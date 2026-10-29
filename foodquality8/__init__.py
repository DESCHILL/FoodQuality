"""FoodQuality public Python API."""
from importlib import import_module
from pathlib import Path
from .dataset.core import Dataset


def module_path(name):
    """Resolve the implementation for cache invalidation and diagnostics."""
    return Path(import_module("foodquality8." + name).__file__)


__all__ = ["Dataset", "module_path"]
