# src/selectors/shap.py
"""
Non-linear feature selection via SHAP values on RandomForestRegressor.

NOTE: Not used in current pipeline — retained for use in the modelling
notebook where computational resources are less constrained.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseSelector
from ..utils import build_storm_weights


class SHAPSelector(BaseSelector):
    """
    Non-linear feature selection via SHAP values on a lightweight
    RandomForestRegressor.

    SHAP TreeExplainer does not shuffle feature values — temporal structure
    preserved. Mean |SHAP| produces more stable rankings under collinearity
    than split-based importance (VIF 159–169 for neutron monitor lags).

    NOTE: Not used in current pipeline — retained for modelling notebook.

    Parameters
    ----------
    feature_cols   : list of str
    storm_thr      : float — storm threshold in nT (default -50)
    n_estimators   : int — RF trees (default 100)
    n_shap_samples : int — rows for SHAP computation (default 10_000)
    random_state   : int (default 42)
    n_jobs         : int (default -1)
    """

    def __init__(
        self,
        feature_cols   : list,
        storm_thr      : float = -50,
        n_estimators   : int   = 100,
        n_shap_samples : int   = 10_000,
        random_state   : int   = 42,
        n_jobs         : int   = -1,
    ):
        super().__init__(feature_cols, storm_thr)
        self.n_estimators   = n_estimators
        self.n_shap_samples = n_shap_samples
        self.random_state   = random_state
        self.n_jobs         = n_jobs
        self.shap_values_   = None
        self.importances_   = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SHAPSelector":
        import shap

        valid        = y.notna()
        X_fit, y_fit = X.loc[valid, self.feature_cols], y.loc[valid]
        weights      = build_storm_weights(y_fit, self.storm_thr)

        rf = RandomForestRegressor(
            n_estimators = self.n_estimators,
            random_state = self.random_state,
            n_jobs       = self.n_jobs,
        )
        rf.fit(X_fit, y_fit, sample_weight=weights)

        rng    = np.random.default_rng(self.random_state)
        n      = min(self.n_shap_samples, len(X_fit))
        idx    = rng.choice(len(X_fit), size=n, replace=False)
        X_shap = X_fit.iloc[idx]

        explainer         = shap.TreeExplainer(rf)
        shap_vals         = explainer.shap_values(X_shap, check_additivity=False)
        self.shap_values_ = shap_vals

        self.importances_ = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=self.feature_cols,
        )
        self.selected_ = list(
            self.importances_[self.importances_ > 0].index
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return mean |SHAP| table sorted descending."""
        return (
            self.importances_
            .rename("mean_abs_shap")
            .to_frame()
            .assign(selected=lambda d: d.index.isin(self.selected_))
            .sort_values("mean_abs_shap", ascending=False)
        )