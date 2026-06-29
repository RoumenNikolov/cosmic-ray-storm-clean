# src/estimators/xgboost_dst_v2.py
"""
XGBoostDstV2 — XGBoost with intensity-proportional storm weighting.

Inherits all parameters and predict() from XGBoostDst.
Overrides only fit() to use build_storm_weights_v2() instead of
build_storm_weights().

Motivation
----------
The original inverse-frequency weighting (w ≈ 21.3) combined with
low learning rate (0.005) and many estimators (500) caused the model
to predict only negative Dst values — the loss function was dominated
by storm hours and the model overfitted to the storm regime.

build_storm_weights_v2() uses intensity-proportional weights:
    w = clip(|Dst| / 50, 1.0, 10.0)

This preserves storm sensitivity while preventing extreme events
from dominating the loss:
    Dst = -50  nT → w = 1.0
    Dst = -100 nT → w = 2.0
    Dst = -200 nT → w = 4.0
    Dst = -422 nT → w = 8.44 (Halloween 2003, capped at 10.0)

References
----------
[BUR75] Burton et al. (1975) — storm threshold -50 nT
[KIS25] Kisvárdai et al. (2025) — storm weighting motivation
[CHE16] Chen & Guestrin (2016) — XGBoost
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .xgboost_dst import XGBoostDst
from ..utils import build_storm_weights_v2


class XGBoostDstV2(XGBoostDst):
    """
    XGBoost regressor with intensity-proportional storm weighting.

    Inherits all parameters and predict() from XGBoostDst.
    Overrides only fit() to use build_storm_weights_v2().

    Parameters
    ----------
    Same as XGBoostDst.
    """

    def fit(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: "pd.Series | np.ndarray",
    ) -> "XGBoostDstV2":
        """
        Fit XGBoost with intensity-proportional storm sample weighting.

        Storm hours receive weight w = clip(|y| / 50, 1.0, 10.0).
        Quiet hours receive unit weight.

        Parameters
        ----------
        X : DataFrame or array — features
        y : array-like — target Dst(t+h)

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f'f{i}' for i in range(X.shape[1])]

        y_arr   = np.asarray(y, dtype=np.float64).ravel()
        weights = build_storm_weights_v2(
            pd.Series(y_arr), storm_thr=self.storm_thr
        )

        self.model_ = XGBRegressor(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            learning_rate    = self.learning_rate,
            subsample        = self.subsample,
            colsample_bytree = self.colsample_bytree,
            reg_alpha        = self.reg_alpha,
            reg_lambda       = self.reg_lambda,
            random_state     = self.random_state,
            n_jobs           = self.n_jobs,
            verbosity        = 0,
        )
        self.model_.fit(X, y_arr, sample_weight=weights)
        return self
