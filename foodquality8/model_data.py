"""Compatibility entry point; implementation lives in dataset."""
import importlib
import sys

_module = importlib.import_module("foodquality8.dataset.model_data")
sys.modules[__name__] = _module
