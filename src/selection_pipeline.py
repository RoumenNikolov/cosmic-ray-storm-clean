# src/selection_pipeline.py
"""
Feature selection pipeline for the cosmic ray storm prediction project.

Orchestrates two independent feature selection methods (LASSO, ExtraTrees)
across all forecast horizons and consolidates the results via majority vote.

Classes
-------
FeatureSelectionPipeline
    Runs LassoSelector and ExtraTreesSelector for each horizon in K_HORIZONS.
    A feature is included in SELECTED_FEATURES if it survives in at least
    min_votes methods at any horizon (union across horizons).

    Two modes are supported:
    - min_votes=2 (intersection): feature must survive both LASSO and ExtraTrees
    - min_votes=1 (union): feature survives if in at least one method

Notes
-----
Both selectors use storm sample weights (1/storm_fraction for
dst < storm_thr) to prevent suppression of Forbush Decrease signals
concentrated in storm periods [KIS25].

LassoSelector uses TimeSeriesSplit internally for alpha selection —
temporally correct for lag features. ExtraTreesSelector does not use
sequential CV — sample_weight handles storm emphasis directly.

Fitted selector objects are stored in fitted_selectors_ for downstream
use (e.g. visualisation) without refitting.
"""

import pandas as pd

from .selectors import LassoSelector, ExtraTreesSelector
from .utils import selection_summary


class FeatureSelectionPipeline:
    """
    Orchestrates LASSO and ExtraTrees feature selection across all horizons.

    Parameters
    ----------
    feature_cols   : list of str — full feature column names
    k_horizons     : list of int — forecast horizons in hours
    storm_thr      : float — storm threshold in nT (default -50)
    lasso_splits   : int — TimeSeriesSplit folds for LassoCV (default 5)
    et_estimators  : int — ExtraTrees n_estimators (default 100)
    et_threshold_q : float — ExtraTrees importance quantile threshold (default 0.50)
    min_votes      : int — minimum methods a feature must survive (default 2)
                     2 = intersection (strict), 1 = union (liberal)
    random_state   : int (default 42)

    Attributes
    ----------
    results_           : dict — {method: {horizon: [selected features]}}
    fitted_selectors_  : dict — {method: {horizon: fitted selector object}}
    summary_           : DataFrame — features × (method, horizon) boolean matrix
    selected_          : list — final SELECTED_FEATURES after majority vote
    vote_counts_       : Series — number of methods each feature survived
    """

    def __init__(
        self,
        feature_cols   : list,
        k_horizons     : list,
        storm_thr      : float = -50,
        lasso_splits   : int   = 5,
        et_estimators  : int   = 100,
        et_threshold_q : float = 0.50,
        min_votes      : int   = 2,
        random_state   : int   = 42,
    ):
        self.feature_cols   = feature_cols
        self.k_horizons     = k_horizons
        self.storm_thr      = storm_thr
        self.lasso_splits   = lasso_splits
        self.et_estimators  = et_estimators
        self.et_threshold_q = et_threshold_q
        self.min_votes      = min_votes
        self.random_state   = random_state

        self.results_          = {'lasso': {}, 'extra_trees': {}}
        self.fitted_selectors_ = {'lasso': {}, 'extra_trees': {}}
        self.summary_          = None
        self.selected_         = None
        self.vote_counts_      = None

    def fit(self, X: pd.DataFrame, feat: pd.DataFrame) -> "FeatureSelectionPipeline":
        """
        Run LassoSelector and ExtraTreesSelector for each horizon.

        Parameters
        ----------
        X    : DataFrame — scaled features (train segment only)
        feat : DataFrame — full feat DataFrame containing dst_target_{k}h columns

        Returns
        -------
        self
        """
        for k in self.k_horizons:
            target_col = f"dst_target_{k}h"
            y          = feat.loc[X.index, target_col]

            print(f"\n── Horizon {k}h ──────────────────────────────────")

            # 1. LASSO
            print(f"  [1/2] LassoCV...")
            lasso = LassoSelector(
                feature_cols = self.feature_cols,
                storm_thr    = self.storm_thr,
                n_splits     = self.lasso_splits,
            ).fit(X, y)
            self.results_['lasso'][k]          = lasso.get_selected()
            self.fitted_selectors_['lasso'][k] = lasso
            print(f"        alpha={lasso.alpha_:.6f}  "
                  f"retained={len(lasso.get_selected())}/{len(self.feature_cols)}")

            # 2. ExtraTrees
            print(f"  [2/2] ExtraTreesSelector...")
            et = ExtraTreesSelector(
                feature_cols = self.feature_cols,
                storm_thr    = self.storm_thr,
                n_estimators = self.et_estimators,
                threshold_q  = self.et_threshold_q,
                random_state = self.random_state,
            ).fit(X, y)
            self.results_['extra_trees'][k]          = et.get_selected()
            self.fitted_selectors_['extra_trees'][k] = et
            print(f"        retained={len(et.get_selected())}/{len(self.feature_cols)}")

        self._consolidate()
        return self

    def _consolidate(self):
        """
        Majority vote consolidation across methods and horizons.

        A feature is included in SELECTED_FEATURES if it survives
        in at least min_votes methods at ANY horizon.
        """
        self.summary_ = selection_summary(self.results_, self.feature_cols)

        vote_counts = pd.Series(0, index=self.feature_cols)

        for k in self.k_horizons:
            cols_k      = [c for c in self.summary_.columns if c.endswith(f"_h{k}")]
            votes_k     = self.summary_[cols_k].sum(axis=1)
            vote_counts = vote_counts.combine(votes_k, max)

        self.vote_counts_ = vote_counts
        self.selected_    = list(
            vote_counts[vote_counts >= self.min_votes].index
        )

    def get_selected(self) -> list:
        """Return final SELECTED_FEATURES after majority vote."""
        if self.selected_ is None:
            raise RuntimeError("fit() must be called before get_selected().")
        return self.selected_

    def print_summary(self):
        """Print vote counts and final selected features."""
        print(f"\n{'='*55}")
        print(f"Feature Selection Summary  (min_votes={self.min_votes})")
        print(f"{'='*55}")
        print(f"\n{'Feature':>30} {'Max votes':>10} {'Selected':>10}")
        print("─" * 55)
        for feature, votes in self.vote_counts_.sort_values(ascending=False).items():
            selected = "✓" if feature in self.selected_ else "✗"
            print(f"{feature:>30} {int(votes):>10} {selected:>10}")
        print(f"\nTotal selected : {len(self.selected_)} / {len(self.feature_cols)}")
        print(f"Selected       : {self.selected_}")