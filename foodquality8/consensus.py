"""Compatibility entry point; implementation lives in ai_detection."""
import importlib
import sys

_module = importlib.import_module("foodquality8.ai_detection.consensus")
sys.modules[__name__] = _module
