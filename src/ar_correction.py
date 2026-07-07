# src/ar_correction.py
"""
Leakage-free AR(2) residual correction for the Operational Forecasting Model stage.

Extracted from cosmic_ray_storm_prediction_ML.ipynb, where the same fit-residuals-then-AR(2)
pattern was repeated inline five times (once per model configuration: LightGBM V1, XGBoost V2,
LightGBM V2, XGBoost V1, MODEL_A), and the walk-forward correction itself (run_hybrid) was
defined once but depended on notebook-level globals (feat_data, y_train, STORM_THR) rather than
being a self-contained function.

Functions
---------
fit_ar2_correction(model, X_train, y_train, label)
    Fit AR(2) on a model's training residuals; print diagnostics (Durbin-Watson, coefficients).

run_hybrid(model, feat_data, seg_mask, seg_name, feature_set, phi1, phi2, c, y_train,
           storm_thr, horizon=7, delay=1)
    Leakage-free walk-forward AR(2) correction, evaluated on one segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.stattools import durbin_watson

from .evaluate import compute_metrics


def fit_ar2_correction(model, X_train: pd.DataFrame, y_train: pd.Series, label: str):
    """
    Fit AR(2) on a model's training residuals; print diagnostics.

    Parameters
    ----------
    model   : fitted estimator with .predict()
    X_train : DataFrame — features matching the segment y_train was computed on
    y_train : Series — target values for the same segment
    label   : str — used in the printed diagnostic line (e.g. 'XGBoost V1 (no dyn)')

    Returns
    -------
    (c, phi1, phi2) : tuple[float, float, float] — AR(2) constant and coefficients
    """
    resid = y_train.values - model.predict(X_train)
    ar = AutoReg(resid, lags=2).fit()
    c, phi1, phi2 = ar.params[0], ar.params[1], ar.params[2]
    print(f'{label}: DW={durbin_watson(resid):.4f}, c={c:.4f}, φ1={phi1:.4f}, φ2={phi2:.4f}')
    return c, phi1, phi2


def align_hybrid_to_window(preds, base_idx, feat_data, window_start, window_end, lag):
    """
    Align AR(2)-hybrid predictions to a specific datetime window, for plotting.

    run_hybrid() internally trims its output by `lag+1` rows relative to the
    original evaluation index (the leakage-free walk-forward correction can only
    start once enough residual history exists). This repeats that same trimming
    to recover the correct datetimes for a subset of `preds`, then filters to a
    window — used, for instance, to zoom into a single storm event.

    Parameters
    ----------
    preds        : array — hybrid predictions, as returned by run_hybrid()
    base_idx     : pd.Index — the original (untrimmed) evaluation index that
                   `preds` was computed from (e.g. test_active_idx)
    feat_data    : DataFrame — full feature matrix (for datetime lookup)
    window_start : pd.Timestamp — window lower bound
    window_end   : pd.Timestamp — window upper bound
    lag          : int — horizon + delay, matching the value passed to run_hybrid()

    Returns
    -------
    (window_preds, window_datetime) : tuple[np.ndarray, np.ndarray]
    """
    trimmed_idx = base_idx[lag + 1:]
    trimmed_datetime = feat_data.loc[trimmed_idx, 'datetime'].values
    window_mask = (trimmed_datetime >= window_start) & (trimmed_datetime <= window_end)
    return preds[window_mask], trimmed_datetime[window_mask]


def run_hybrid(
    model,
    feat_data: pd.DataFrame,
    seg_mask,
    seg_name: str,
    feature_set: list,
    phi1: float,
    phi2: float,
    c: float,
    y_train: pd.Series,
    storm_thr: float,
    horizon: int = 7,
    delay: int = 1,
    dst_col: str = 'dst',
):
    """
    Leakage-free walk-forward AR(2) correction.

    A residual computed at row t reflects the target realised `horizon` hours later;
    with an additional publication delay, it only becomes usable at lag = horizon + delay.

    Parameters
    ----------
    model       : fitted estimator with .predict()
    feat_data   : DataFrame — full feature matrix
    seg_mask    : boolean Series — evaluation segment (e.g. val_main_mask)
    seg_name    : str — segment label for the printed metrics line
    feature_set : list of str — feature columns matching what `model` was fit on
    phi1, phi2  : float — AR(2) coefficients (from fit_ar2_correction)
    c           : float — AR(2) constant (from fit_ar2_correction)
    y_train     : array-like — training target series, for MASE denominator in compute_metrics
    storm_thr   : float — storm threshold in nT
    horizon     : int — forecast horizon in hours (default 7)
    delay       : int — publication delay in hours (default 1)
    dst_col     : str — column name for current Dst, used as persistence baseline (default 'dst')

    Returns
    -------
    (metrics, y_hybrid_eval, y_true_eval) : tuple[dict, np.ndarray, np.ndarray]
    """
    target_col = f'dst_target_{horizon}h'
    y_true = feat_data.loc[seg_mask, target_col].dropna()
    idx    = y_true.index
    y_pred = model.predict(feat_data.loc[idx, feature_set])
    resid  = y_true.values - y_pred

    lag = horizon + delay  # earliest genuinely available residual offset
    y_hybrid = y_pred.copy()

    for t in range(lag + 1, len(resid)):
        e_t_lag   = resid[t - lag]
        e_t_lag_1 = resid[t - lag - 1]
        ar_correction = c + phi1 * e_t_lag + phi2 * e_t_lag_1
        y_hybrid[t] += ar_correction

    y_true_eval    = y_true.values[lag+1:]
    y_hybrid_eval  = y_hybrid[lag+1:]
    y_persist_eval = feat_data.loc[idx, dst_col].values[lag+1:]

    m = compute_metrics(
        y_true=y_true_eval, y_pred=y_hybrid_eval, y_train=y_train,
        y_persist=y_persist_eval, storm_thr=storm_thr, horizon=horizon,
    )
    print(f'{seg_name:<12} RMSE={m["rmse"]:6.2f} | StormRMSE={m["storm_rmse"]:6.2f} | R²={m["r2"]:6.4f}')
    return m, y_hybrid_eval, y_true_eval
