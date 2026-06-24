# src/selectors/lasso.py
"""
Linear feature selection via weighted LassoCV with TimeSeriesSplit
cross-validation. Identifies features with non-zero linear contribution
at the cross-validated penalty level. Storm hours are upweighted by
inverse storm fraction to prevent suppression of Forbush Decrease signals
(Kisvárdai et al. 2025 [KIS25]).
"""

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseSelector
from ..utils import build_storm_weights


class LassoSelector(BaseSelector):
    """
    Linear feature selection via weighted LassoCV.

    Features whose coefficients fall below threshold=1e-5 after
    cross-validated L1 regularisation are excluded. The threshold
    eliminates floating-point near-zero artefacts from the numerical
    optimisation rather than ranking features by importance.

    Parameters
    ----------
    feature_cols : list of str
    storm_thr    : float — storm threshold in nT (default -50)
    n_splits     : int — TimeSeriesSplit folds for alpha selection (default 5)
    threshold    : float — coefficient threshold (default 1e-5)
    max_iter     : int — LassoCV max iterations (default 10_000)
    """

    def __init__(
        self,
        feature_cols : list,
        storm_thr    : float = -50,
        n_splits     : int   = 5,
        threshold    : float = 1e-5,
        max_iter     : int   = 10_000,
    ):
        super().__init__(feature_cols, storm_thr)
        self.n_splits  = n_splits
        self.threshold = threshold
        self.max_iter  = max_iter
        self.lasso_    = None
        self.alpha_    = None
        self.coef_     = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoSelector":
        valid        = y.notna()
        X_fit, y_fit = X.loc[valid], y.loc[valid]
        weights      = build_storm_weights(y_fit, self.storm_thr)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.lasso_ = LassoCV(
            cv            = tscv,
            fit_intercept = True,
            max_iter      = self.max_iter,
            n_jobs        = -1,
        )
        self.lasso_.fit(X_fit[self.feature_cols], y_fit, sample_weight=weights)

        self.alpha_ = self.lasso_.alpha_
        self.coef_  = pd.Series(self.lasso_.coef_, index=self.feature_cols)

        selector       = SelectFromModel(self.lasso_, threshold=self.threshold, prefit=True)
        support        = selector.get_support()
        self.selected_ = [f for f, s in zip(self.feature_cols, support) if s]

        return self

    def summary(self) -> pd.DataFrame:
        """Return coefficient table sorted by absolute value."""
        return (
            self.coef_
            .rename("coefficient")
            .to_frame()
            .assign(selected=lambda d: d.index.isin(self.selected_))
            .sort_values("coefficient", key=abs, ascending=False)
        )