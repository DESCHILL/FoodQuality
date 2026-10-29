"""Compatibility entry point; implementation lives in ai_quality."""
import importlib
import sys

_module = importlib.import_module("foodquality8.ai_quality.food9")
sys.modules[__name__] = _module
