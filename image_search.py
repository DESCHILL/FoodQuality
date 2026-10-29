"""Compatibility import for the dataset collector."""
import sys
from foodquality8.dataset import image_search as _module
sys.modules[__name__] = _module
