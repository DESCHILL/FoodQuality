"""Compatibility entry point; implementation lives in ai_quality."""
import importlib
import sys

_module = importlib.import_module("foodquality8.ai_quality.training")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
