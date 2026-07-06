# src/estimators/lightgbm_dst.py
"""
LightGBM wrapper for geomagnetic storm prediction.

Mirror of XGBoostDst — adds two things above a bare LGBMRegressor:
1. Storm sample weighting via build_storm_weights() — upweights storm hours
   (Dst < storm_thr) by 1/storm_fraction to counteract class imbalance.
2. feature_names_in_ — stored at fit time for downstream SHAP analysis.

LightGBM uses a leaf-wise tree-growing strategy, in contrast to XGBoost's
default level-wise approach. This may produce different generalisation
behaviour on storm-dominated data and is the primary motivation for
including it as an independent comparator.

All hyperparameters are passed through to LGBMRegressor and can be set
via Pipeline + GridSearchCV in the modelling notebook.

Usage
-----
pipe = Pipeline([
    ('model', LightGBMDst(n_estimators=500, max_depth=5))
])
pipe.fit(X_train[FEATURE_COLS], y_train)
y_pred = pipe.predict(X_val[FEATURE_COLS])

References
----------
[KIS25] Kisvárdai et al. (2025) — storm weighting motivation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor

from ..utils import build_storm_weights


class LightGBMDst(BaseEstimator, RegressorMixin):
    """
    LightGBM regressor with storm sample weighting.

    Parameters
    ----------
    n_estimators   : int   — number of boosting rounds (default 500)
    max_depth      : int   — maximum tree depth, -1 = no limit (default 5)
    learning_rate  : float — step size shrinkage (default 0.05)
    num_leaves     : int   — maximum number of leaves per tree (default 31)
    subsample      : float — row subsampling ratio (default 0.8)
    subsample_freq : int   — bagging frequency in boosting rounds; must be > 0
                     for `subsample` to have any effect — LightGBM's own default
                     of 0 disables bagging regardless of `subsample` (default 1)
    colsample_bytree: float — feature subsampling ratio (default 0.8)
    reg_alpha      : float — L1 regularisation (default 0.0)
    reg_lambda     : float — L2 regularisation (default 1.0)
    storm_thr      : float — storm threshold in nT for sample weighting (default -50)
    random_state   : int   (default 42)
    n_jobs         : int   (default -1)

    Attributes
    ----------
    model_             : LGBMRegressor — fitted underlying estimator
    feature_names_in_  : list of str — feature names from fit DataFrame
    """

    def __init__(
        self,
        n_estimators    : int   = 500,
        max_depth       : int   = 5,
        learning_rate   : float = 0.05,
        num_leaves      : int   = 31,
        subsample       : float = 0.8,
        subsample_freq  : int   = 1,
        colsample_bytree: float = 0.8,
        reg_alpha       : float = 0.0,
        reg_lambda      : float = 1.0,
        storm_thr       : float = -50.0,
        random_state    : int   = 42,
        n_jobs          : int   = -1,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.num_leaves       = num_leaves
        self.subsample        = subsample
        self.subsample_freq   = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.storm_thr        = storm_thr
        self.random_state     = random_state
        self.n_jobs           = n_jobs

        self.model_            = None
        self.feature_names_in_ = None

    def fit(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: "pd.Series | np.ndarray",
    ) -> "LightGBMDst":
        """
        Fit LightGBM with storm sample weighting.

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
        weights = build_storm_weights(
            pd.Series(y_arr), storm_thr=self.storm_thr
        )

        self.model_ = LGBMRegressor(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            learning_rate    = self.learning_rate,
            num_leaves       = self.num_leaves,
            subsample        = self.subsample,
            subsample_freq   = self.subsample_freq,
            colsample_bytree = self.colsample_bytree,
            reg_alpha        = self.reg_alpha,
            reg_lambda       = self.reg_lambda,
            random_state     = self.random_state,
            n_jobs           = self.n_jobs,
            verbose          = -1,
        )
        self.model_.fit(X, y_arr, sample_weight=weights)
        return self

    def predict(
        self,
        X: "pd.DataFrame | np.ndarray",
    ) -> np.ndarray:
        """
        Predict Dst(t+h).

        Parameters
        ----------
        X : DataFrame or array — features

        Returns
        -------
        np.ndarray of shape (n_samples,) — predictions in nT
        """
        if self.model_ is None:
            raise RuntimeError("fit() must be called before predict().")
        return self.model_.predict(X)

    def get_booster(self):
        """Return underlying LightGBM Booster for SHAP analysis."""
        if self.model_ is None:
            raise RuntimeError("fit() must be called before get_booster().")
        return self.model_.booster_
