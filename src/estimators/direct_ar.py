# src/estimators/direct_ar.py
"""
Direct Autoregressive Baseline for geomagnetic storm prediction.

Models the h-step-ahead Dst as a linear function of the current Dst:

    D_st(t+h) = alpha_h * D_st(t) + beta_h

One instance per forecast horizon h. Unlike a classical AR(1) model,
which propagates one step at a time, this is a direct forecasting model —
it predicts Dst(t+h) directly without iterating through intermediate steps.

The fitted alpha_h coefficient has a physical interpretation: it approximates
the fraction of the current ring current disturbance that persists after h hours,
consistent with the Burton et al. (1975) [BUR75] exponential decay model with
relaxation time tau ~ 7-8h.

This baseline occupies the second rung in the predictability hierarchy:

    Persistence           — zero complexity, no learning
    DirectARBaseline      — linear memory of Dst
    XGBoost (OMNI)        — nonlinear solar wind forcing
    XGBoost (OMNI + δn)   — H_gain hypothesis test

References
----------
[BUR75] Burton, R. K., McPherron, R. L., & Russell, C. T. (1975).
        An empirical relationship between interplanetary conditions and Dst.
        Journal of Geophysical Research, 80(31), 4204-4214.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class DirectARBaseline(BaseEstimator, RegressorMixin):
    """
    Direct autoregressive baseline — single-feature OLS per horizon.

    Fits D_st(t+h) = alpha_h * D_st(t) + beta_h using ordinary least squares.
    Intentionally minimal: only the current Dst is used as a predictor.
    Adding more lags would make this a competitive ML model rather than a baseline.

    Parameters
    ----------
    dst_col : str — column name for current Dst in input DataFrame (default 'dst')

    Attributes
    ----------
    alpha_ : float — fitted slope (ring current decay fraction)
    beta_  : float — fitted intercept
    model_ : LinearRegression — underlying sklearn estimator
    """

    def __init__(self, dst_col: str = 'dst'):
        self.dst_col = dst_col

    def fit(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: "pd.Series | np.ndarray",
        sample_weight: "np.ndarray | None" = None,
    ) -> "DirectARBaseline":
        """
        Fit OLS on current Dst only.

        Parameters
        ----------
        X             : DataFrame or array — must contain dst_col if DataFrame
        y             : array-like — target Dst(t+h)
        sample_weight : optional sample weights (passed to LinearRegression.fit)

        Returns
        -------
        self
        """
        X_dst = self._extract_dst(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        # Drop NaN rows
        mask  = np.isfinite(X_dst.ravel()) & np.isfinite(y_arr)
        X_fit = X_dst[mask].reshape(-1, 1)
        y_fit = y_arr[mask]
        sw    = sample_weight[mask] if sample_weight is not None else None

        self.model_ = LinearRegression()
        self.model_.fit(X_fit, y_fit, sample_weight=sw)

        self.alpha_ = float(self.model_.coef_[0])
        self.beta_  = float(self.model_.intercept_)

        return self

    def predict(
        self,
        X: "pd.DataFrame | np.ndarray",
    ) -> np.ndarray:
        """
        Predict D_st(t+h) = alpha_h * D_st(t) + beta_h.

        Parameters
        ----------
        X : DataFrame or array — must contain dst_col if DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,) — predictions in nT
        """
        X_dst = self._extract_dst(X).reshape(-1, 1)
        return self.model_.predict(X_dst)

    def _extract_dst(
        self,
        X: "pd.DataFrame | np.ndarray",
    ) -> np.ndarray:
        """Extract Dst column from DataFrame or pass through array."""
        if isinstance(X, pd.DataFrame):
            if self.dst_col not in X.columns:
                raise ValueError(
                    f"DirectARBaseline: column '{self.dst_col}' not found in X. "
                    f"Available columns: {list(X.columns)}"
                )
            return np.asarray(X[self.dst_col], dtype=np.float64)
        return np.asarray(X, dtype=np.float64).ravel()

    def summary(self) -> dict:
        """
        Return fitted coefficients as a dict.

        Returns
        -------
        dict with keys: alpha, beta
            alpha — ring current decay fraction at this horizon
            beta  — intercept (baseline Dst offset)
        """
        if self.model_ is None:
            raise RuntimeError("fit() must be called before summary().")
        return {'alpha': self.alpha_, 'beta': self.beta_}
