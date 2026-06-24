# src/__init__.py
"""
Feature selection package for the cosmic ray storm prediction project.
"""

from .selectors import LassoSelector, ExtraTreesSelector, MISelector, SHAPSelector
from .selection_pipeline import FeatureSelectionPipeline
from .selection_visualization import plot_lasso_vs_et, plot_et_heatmap
from .utils import build_storm_weights, selection_summary
from .split_visualization import plot_dst_split_distribution

__all__ = [
    "LassoSelector",
    "ExtraTreesSelector",
    "MISelector",
    "SHAPSelector",
    "FeatureSelectionPipeline",
    "plot_lasso_vs_et",
    "plot_et_heatmap",
    "build_storm_weights",
    "selection_summary",
]