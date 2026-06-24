# src/selectors/mi.py
"""
Non-linear feature selection via Mutual Information regression.

NOTE: Not used in current pipeline — sample_weight not supported in
mutual_info_regression in this sklearn build. Retained for future use.
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from .base import BaseSelector
from ..utils import build_storm_weights


class MISelector(BaseSelector):
    """
    Non-linear feature selection via Mutual Information regression.

    Features are selected if they satisfy both:
    - MI score >= 25th percentile of non-zero MI scores (noise floor)
    - MI score >= 50th percentile of all MI scores (top half)

    NOTE: Not used in current pipeline — sample_weight not supported
    in mutual_info_regression in this sklearn build.

    Parameters
    ----------
    feature_cols  : list of str
    storm_thr     : float — storm threshold in nT (default -50)
    random_state  : int (default 42)
    """

    def __init__(
        self,
        feature_cols : list,
        storm_thr    : float = -50,
        random_state : int   = 42,
    ):
        super().__init__(feature_cols, storm_thr)
        self.random_state = random_state
        self.mi_scores_   = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MISelector":
        valid        = y.notna()
        X_fit, y_fit = X.loc[valid], y.loc[valid]
        weights      = build_storm_weights(y_fit, self.storm_thr)

        scores = mutual_info_regression(
            X_fit[self.feature_cols],
            y_fit,
            random_state  = self.random_state,
            sample_weight = weights,
        )
        self.mi_scores_ = pd.Series(scores, index=self.feature_cols)

        nonzero        = self.mi_scores_[self.mi_scores_ > 0]
        p25            = nonzero.quantile(0.25) if len(nonzero) > 0 else 0
        p50            = self.mi_scores_.quantile(0.50)

        self.selected_ = list(
            self.mi_scores_[
                (self.mi_scores_ >= p25) & (self.mi_scores_ >= p50)
            ].index
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return MI score table sorted descending."""
        return (
            self.mi_scores_
            .rename("mi_score")
            .to_frame()
            .assign(selected=lambda d: d.index.isin(self.selected_))
            .sort_values("mi_score", ascending=False)
        )