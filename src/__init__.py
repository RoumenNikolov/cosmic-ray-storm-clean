# src/__init__.py
"""
Feature selection package for the cosmic ray storm prediction project.

Modules
-------
feature_selector
    BaseSelector, LassoSelector, RFECVSelector, SHAPSelector

selection_pipeline
    FeatureSelectionPipeline

utils
    build_storm_weights, selection_summary
"""

from .feature_selector import LassoSelector, RFECVSelector, SHAPSelector
from .selection_pipeline import FeatureSelectionPipeline
from .utils import build_storm_weights, selection_summary

__all__ = [
    "LassoSelector",
    "RFECVSelector",
    "SHAPSelector",
    "FeatureSelectionPipeline",
    "build_storm_weights",
    "selection_summary",
]