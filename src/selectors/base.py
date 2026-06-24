# src/selectors/base.py
"""
Abstract base class for all feature selectors.
"""

import pandas as pd
from abc import ABC, abstractmethod


class BaseSelector(ABC):
    """
    Abstract base class for feature selectors.

    All selectors share a common interface: fit() on training data,
    get_selected() to retrieve surviving features.

    Parameters
    ----------
    feature_cols : list of str — full feature column names
    storm_thr    : float — storm threshold in nT (default -50)
    """

    def __init__(self, feature_cols: list, storm_thr: float = -50):
        self.feature_cols = feature_cols
        self.storm_thr    = storm_thr
        self.selected_    = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSelector":
        """Fit the selector on training data."""
        pass

    def get_selected(self) -> list:
        """
        Return list of selected feature names.

        Raises
        ------
        RuntimeError if fit() has not been called.
        """
        if self.selected_ is None:
            raise RuntimeError("fit() must be called before get_selected().")
        return self.selected_