"""Compatibility entry point; implementation lives in ai_quality."""
import importlib
import sys

_module = importlib.import_module("foodquality8.ai_quality.model_report")
sys.modules[__name__] = _module
