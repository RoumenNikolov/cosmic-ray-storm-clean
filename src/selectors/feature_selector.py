# src/feature_selector.py
"""
Backward compatibility shim — imports from new selectors/ package.
Do not add new selectors here.
"""

from .selectors import LassoSelector, ExtraTreesSelector, MISelector, SHAPSelector

__all__ = [
    "LassoSelector",
    "ExtraTreesSelector",
    "MISelector",
    "SHAPSelector",
]