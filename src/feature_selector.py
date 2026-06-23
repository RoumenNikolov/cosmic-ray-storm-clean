# src/feature_selector.py
"""
Feature selector classes for the cosmic ray storm prediction project.

Three independent feature selection methods are implemented, each targeting
a different aspect of the feature-target relationship:

Classes
-------
BaseSelector
    Abstract base class defining the common interface for all selectors.
    Subclasses must implement fit() and get_selected().

LassoSelector
    Linear feature selection via weighted LassoCV with TimeSeriesSplit
    cross-validation. Identifies features with non-zero linear contribution
    at the cross-validated penalty level. Storm hours are upweighted by
    inverse storm fraction to prevent suppression of Forbush Decrease signals
    (Kisvárdai et al. 2025 [KIS25]).

MISelector
    Non-linear feature selection via Mutual Information regression. Captures
    dependencies that LASSO misses by definition. Features are selected if
    their MI score exceeds the 25th percentile of non-zero scores AND falls
    in the top 50% of all scores — a dual criterion that avoids both
    near-zero noise and arbitrary absolute thresholds.

SHAPSelector
    Non-linear feature selection via SHAP values on a lightweight
    RandomForestRegressor. Preferred over permutation importance for time
    series data because SHAP TreeExplainer does not shuffle feature values
    — it preserves the temporal structure of lag features. Preferred over
    native RF importance due to documented multicollinearity among neutron
    monitor lag features (VIF 159-169) — mean |SHAP| produces more stable
    rankings under collinearity than split-based importance.

Notes
-----
All selectors accept sample_weight in fit() to ensure storm periods
receive appropriate emphasis during selection. The majority vote
consolidation is handled by FeatureSelectionPipeline in selection_pipeline.py.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from .utils import build_storm_weights


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
        """
        Fit LassoCV with storm-weighted samples.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features) — scaled features
        y : Series of shape (n_samples,) — regression target

        Returns
        -------
        self
        """
        valid            = y.notna()
        X_fit, y_fit     = X.loc[valid], y.loc[valid]
        weights          = build_storm_weights(y_fit, self.storm_thr)
        valid            = y.notna()
        X_fit, y_fit     = X.loc[valid], y.loc[valid]
        weights          = build_storm_weights(y_fit, self.storm_thr)
        

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


class MISelector(BaseSelector):
    """
    Non-linear feature selection via Mutual Information regression.

    Features are selected if they satisfy both:
    - MI score >= 25th percentile of non-zero MI scores (noise floor)
    - MI score >= 50th percentile of all MI scores (top half)

    This dual criterion avoids arbitrary absolute thresholds while
    excluding near-zero noise features regardless of ranking.

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
        """
        Compute MI scores with storm-weighted samples.

        Parameters
        ----------
        X : DataFrame — scaled features
        y : Series — regression target

        Returns
        -------
        self
        """
        valid            = y.notna()
        X_fit, y_fit     = X.loc[valid], y.loc[valid]
        weights          = build_storm_weights(y_fit, self.storm_thr)

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


class SHAPSelector(BaseSelector):
    """
    Non-linear feature selection via SHAP values on a lightweight
    RandomForestRegressor.

    Preferred over permutation importance for time series data because
    SHAP TreeExplainer does not shuffle feature values — it preserves
    the temporal structure of lag features. Preferred over native RF
    importance due to documented multicollinearity among neutron monitor
    lag features (VIF 159-169) — mean |SHAP| produces more stable
    rankings under collinearity than split-based importance.

    SHAP is computed on a subsample of n_shap_samples rows for speed.
    Mean |SHAP| converges at ~10k rows — ranking is stable beyond this.

    Features with mean |SHAP| > 0 are retained.

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
        """
        Fit RF and compute mean |SHAP| per feature with storm weights.
        SHAP is computed on a subsample of n_shap_samples rows for speed.
        TreeExplainer does not shuffle values — temporal structure preserved.

        Parameters
        ----------
        X : DataFrame — scaled features
        y : Series — regression target

        Returns
        -------
        self
        """
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

        # Subsample for SHAP — mean |SHAP| converges at ~10k rows
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

class RFECVSelector(BaseSelector):
    """
    Temporally-correct feature selection via Recursive Feature Elimination
    with Cross-Validation (RFECV) on LinearRegression.

    Uses TimeSeriesSplit to preserve temporal order during cross-validation.
    Preferred over SHAP-based selection for time series data when computational
    resources are limited — RFECV iteratively eliminates features based on
    cross-validated R², respecting the arrow of time.

    Parameters
    ----------
    feature_cols : list of str
    storm_thr    : float — storm threshold in nT (default -50)
    n_splits     : int — TimeSeriesSplit folds (default 5)
    n_jobs       : int (default -1)
    """

    def __init__(
        self,
        feature_cols : list,
        storm_thr    : float = -50,
        n_splits     : int   = 5,
        n_jobs       : int   = -1,
    ):
        super().__init__(feature_cols, storm_thr)
        self.n_splits  = n_splits
        self.n_jobs    = n_jobs
        self.rfecv_    = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RFECVSelector":
        """
        Fit RFECV with TimeSeriesSplit on training data.

        Note: RFECV does not support sample_weight directly.
        Storm weighting is approximated via oversampling — consistent
        with the MISelector approach.

        Parameters
        ----------
        X : DataFrame — scaled features
        y : Series — regression target

        Returns
        -------
        self
        """
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import TimeSeriesSplit

        valid        = y.notna()
        X_fit, y_fit = X.loc[valid, self.feature_cols], y.loc[valid]
        weights      = build_storm_weights(y_fit, self.storm_thr)

        # Oversample storm hours
        rng       = np.random.default_rng(42)
        n_samples = len(y_fit)
        idx       = rng.choice(n_samples, size=n_samples, p=weights/weights.sum())
        X_over    = X_fit.iloc[idx]
        y_over    = y_fit.iloc[idx]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.rfecv_ = RFECV(
            estimator = LinearRegression(),
            cv        = tscv,
            scoring   = 'r2',
            n_jobs    = self.n_jobs,
        )
        self.rfecv_.fit(X_over, y_over)
        self.selected_ = list(
            np.array(self.feature_cols)[self.rfecv_.support_]
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return feature ranking table sorted by RFECV ranking."""
        ranking = pd.Series(self.rfecv_.ranking_, index=self.feature_cols)
        return (
            ranking
            .rename("ranking")
            .to_frame()
            .assign(selected=lambda d: d.index.isin(self.selected_))
            .sort_values("ranking")
        )