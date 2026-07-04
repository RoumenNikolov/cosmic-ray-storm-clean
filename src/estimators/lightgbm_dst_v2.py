# src/estimators/lightgbm_dst_v2.py
"""
LightGBMDstV2 — LightGBM with intensity-proportional storm weighting.

Inherits all parameters and predict() from LightGBMDst.
Overrides only fit() to use build_storm_weights_v2() instead of
build_storm_weights().

Motivation
----------
Same as XGBoostDstV2 — see xgboost_dst_v2.py for full motivation.

References
----------
[BUR75] Burton et al. (1975) — storm threshold -50 nT
[KIS25] Kisvárdai et al. (2025) — storm weighting motivation
[KE17]  Ke et al. (2017) — LightGBM
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .lightgbm_dst import LightGBMDst
from ..utils import build_storm_weights_v2


class LightGBMDstV2(LightGBMDst):
    """
    LightGBM regressor with intensity-proportional storm weighting.

    Inherits all parameters and predict() from LightGBMDst.
    Overrides only fit() to use build_storm_weights_v2().

    Parameters
    ----------
    Same as LightGBMDst.
    """

    def fit(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: "pd.Series | np.ndarray",
    ) -> "LightGBMDstV2":
        """
        Fit LightGBM with intensity-proportional storm sample weighting.

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

        self.model_ = LGBMRegressor(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            learning_rate    = self.learning_rate,
            num_leaves       = self.num_leaves,
            subsample        = self.subsample,
            colsample_bytree = self.colsample_bytree,
            reg_alpha        = self.reg_alpha,
            reg_lambda       = self.reg_lambda,
            random_state     = self.random_state,
            n_jobs           = self.n_jobs,
            verbose          = -1,
        )
        self.model_.fit(X, y_arr, sample_weight=weights)
        return self
