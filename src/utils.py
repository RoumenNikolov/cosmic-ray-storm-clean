# src/utils.py
"""
Utility functions shared across feature selection methods.

Functions
---------
build_storm_weights(y, storm_thr)
    Compute inverse-frequency sample weights for storm hours.
    Storm hours (dst < storm_thr) are upweighted by 1/storm_fraction
    to counteract the severe class imbalance (~5% storm hours in training set).
    Physical motivation: Kisvárdai et al. (2025) [KIS25] demonstrate that
    neutron monitor correlation with Dst increases from 0.306 to 0.555
    during Forbush Decrease periods — an unweighted fit suppresses features
    whose signal is concentrated in storm periods.

selection_summary(results, feature_cols)
    Build a summary DataFrame of selected features per method and horizon.
    Rows are features, columns are (method, horizon) combinations.
    Values are boolean — True if the feature survived the method at that horizon.
    Used to compute the majority vote in FeatureSelectionPipeline.
"""

import numpy as np
import pandas as pd


def build_storm_weights(y: pd.Series, storm_thr: float = -50) -> np.ndarray:
    """
    Compute inverse-frequency sample weights for storm hours.

    Storm hours (y < storm_thr) are upweighted by 1/storm_fraction
    to counteract class imbalance. Non-storm hours receive weight 1.0.

    Parameters
    ----------
    y         : target Series (dst_target_{k}h)
    storm_thr : storm threshold in nT (default -50)

    Returns
    -------
    weights : np.ndarray of shape (n_samples,)
    """
    storm_mask     = y < storm_thr
    storm_fraction = storm_mask.mean()
    w_storm        = 1.0 / storm_fraction

    weights = np.where(storm_mask, w_storm, 1.0)
    return weights


def selection_summary(results: dict, feature_cols: list) -> pd.DataFrame:
    """
    Build a summary DataFrame of selected features per method and horizon.

    Parameters
    ----------
    results      : dict — {method: {horizon: [selected features]}}
    feature_cols : list — full feature list

    Returns
    -------
    df : DataFrame with features as rows, (method, horizon) as columns
         values are True/False
    """
    records = {}
    for method, horizons in results.items():
        for horizon, selected in horizons.items():
            col = f"{method}_h{horizon}"
            records[col] = {f: f in selected for f in feature_cols}

    return pd.DataFrame(records, index=feature_cols)