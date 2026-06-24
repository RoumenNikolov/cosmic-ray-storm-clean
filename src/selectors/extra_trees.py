# src/selectors/extra_trees.py
"""
Non-linear feature selection via ExtraTreesRegressor feature importances.
Preferred over Mutual Information because ExtraTreesRegressor.fit() accepts
sample_weight directly — storm hours can be upweighted without resampling.
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .base import BaseSelector
from ..utils import build_storm_weights


class ExtraTreesSelector(BaseSelector):
    """
    Non-linear feature selection via ExtraTreesRegressor feature importances.

    Features with importance >= quantile threshold of all importances
    are retained. Storm hours are upweighted by inverse storm fraction
    to prevent suppression of Forbush Decrease signals [KIS25].

    Parameters
    ----------
    feature_cols  : list of str
    storm_thr     : float — storm threshold in nT (default -50)
    n_estimators  : int — number of trees (default 100)
    threshold_q   : float — quantile threshold for importance (default 0.50)
                    0.50 = top 50%, 0.0 = all features with importance > 0
    random_state  : int (default 42)
    n_jobs        : int (default -1)
    """

    def __init__(
        self,
        feature_cols  : list,
        storm_thr     : float = -50,
        n_estimators  : int   = 100,
        threshold_q   : float = 0.50,
        random_state  : int   = 42,
        n_jobs        : int   = -1,
    ):
        super().__init__(feature_cols, storm_thr)
        self.n_estimators  = n_estimators
        self.threshold_q   = threshold_q
        self.random_state  = random_state
        self.n_jobs        = n_jobs
        self.importances_  = None
        self.model_        = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExtraTreesSelector":
        valid        = y.notna()
        X_fit, y_fit = X.loc[valid, self.feature_cols], y.loc[valid]
        weights      = build_storm_weights(y_fit, self.storm_thr)

        self.model_ = ExtraTreesRegressor(
            n_estimators = self.n_estimators,
            random_state = self.random_state,
            n_jobs       = self.n_jobs,
        )
        self.model_.fit(X_fit, y_fit, sample_weight=weights)

        self.importances_ = pd.Series(
            self.model_.feature_importances_,
            index=self.feature_cols,
        )

        threshold      = self.importances_.quantile(self.threshold_q)
        self.selected_ = list(
            self.importances_[self.importances_ >= threshold].index
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return feature importance table sorted descending."""
        return (
            self.importances_
            .rename("importance")
            .to_frame()
            .assign(selected=lambda d: d.index.isin(self.selected_))
            .sort_values("importance", ascending=False)
        )