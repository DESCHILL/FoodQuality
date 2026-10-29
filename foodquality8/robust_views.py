"""Compatibility entry point; implementation lives in ai_quality."""
import importlib
import sys

_module = importlib.import_module("foodquality8.ai_quality.robust_views")
sys.modules[__name__] = _module
