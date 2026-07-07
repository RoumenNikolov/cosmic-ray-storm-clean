# src/runner.py
"""
Generic model training and evaluation runner for the cosmic ray storm
prediction project.

All functions follow a single-responsibility pattern and are designed to
be reused across all model types (Persistence, DirectAR, XGBoost, LightGBM)
without modification. Model-specific behaviour is passed via parameters.

Functions
---------
fit_model(estimator, X_train, y_train)
    Fit any sklearn-compatible estimator on training data. Returns fitted estimator.

print_horizon_metrics(h, metrics, extra=None)
    Print RMSE, Storm RMSE and MASE for one horizon. Optional extra dict
    adds model-specific fields (e.g. alpha, beta for DirectARBaseline).

log_metrics_run(run_name, tags, params, metrics, nested)
    Log a single MLflow run with tags, params and metrics.
    NaN and inf values are excluded from metrics logging.
    Wrapped with mlflow_safe (src.mlflow_tracking) — MLflow failures
    emit a warning instead of raising.

evaluate_model_on_segments(models_dict, eval_segments, feat_data,
            feature_set, y_train, horizon, storm_thr, dst_col)
    Evaluate each model in models_dict on every segment in eval_segments,
    at a single fixed horizon. Pure computation — no MLflow side effects,
    no printing. Returns dict {model_name: {seg_name: metrics_dict}}.

run_horizon(model_name, seg_name, h, model, X_seg, y_true,
            y_train, y_persist, storm_thr, extra_params)
    Predict and evaluate model at one horizon, log nested MLflow child run.
    Calls model.predict() and compute_metrics() directly — no intermediate layer.
    Returns metrics dict.

run_segment(model_name, seg_name, X_seg, y_seg, y_train, y_persist,
            models, storm_thr, k_horizons, extra_params_fn)
    Evaluate model for all horizons in one segment.
    Opens parent MLflow run (via safe_mlflow_run) and delegates to run_horizon().
    Returns dict {horizon: metrics}.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import mlflow

from .evaluate import compute_metrics
from .mlflow_tracking import mlflow_safe, safe_mlflow_run


# ── Training ──────────────────────────────────────────────────────────────────

def fit_model(estimator, X_train, y_train):
    """
    Fit any sklearn-compatible estimator on training data.

    Parameters
    ----------
    estimator : sklearn estimator — unfitted model or Pipeline
    X_train   : DataFrame or array — training features
    y_train   : Series or array — training targets

    Returns
    -------
    fitted estimator
    """
    estimator.fit(X_train, y_train)
    return estimator


# ── Display ───────────────────────────────────────────────────────────────────

def print_horizon_metrics(h: int, metrics: dict, extra: dict = None):
    """
    Print RMSE, Storm RMSE and MASE for one horizon.

    Parameters
    ----------
    h       : int — forecast horizon in hours
    metrics : dict — output of compute_metrics()
    extra   : dict or None — additional model-specific fields to prepend.
              e.g. {'alpha': 0.774, 'beta': -5.2} for DirectARBaseline.
    """
    prefix = ''
    if extra:
        prefix = '  '.join(f'{k}={v:.3f}' for k, v in extra.items()) + ' | '
    print(f'  {prefix}h={h:>2}h | '
          f'RMSE={metrics["rmse"]:6.2f} | '
          f'StormRMSE={metrics["storm_rmse"]:6.2f} | '
          f'MASE={metrics["mase"]:5.3f}')


# ── MLflow logging ────────────────────────────────────────────────────────────

@mlflow_safe
def log_metrics_run(
    run_name : str,
    tags     : dict,
    params   : dict,
    metrics  : dict,
    nested   : bool = False,
):
    """
    Log a single MLflow run with tags, params and metrics.
    NaN and inf values are excluded from metrics logging.
    No-ops (does nothing) if ENABLE_MLFLOW is off (src/config.py) —
    goes through safe_mlflow_run(), same as run_segment() below, rather
    than calling mlflow.start_run() directly.

    Parameters
    ----------
    run_name : str — MLflow run name
    tags     : dict — MLflow tags (model_type, evaluation_set, etc.)
    params   : dict — MLflow params (horizon, storm_thr, feature_set, etc.)
    metrics  : dict — output of compute_metrics()
    nested   : bool — whether this is a nested child run (default False)
    """
    with safe_mlflow_run(run_name=run_name, nested=nested) as run:
        if run is None:
            return
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.log_params(params)
        mlflow.log_metrics({
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and np.isfinite(float(v))
        })


# ── Segment-level evaluation (single horizon) ──────────────────────────────────

def evaluate_model_on_segments(
    models_dict     : dict,
    eval_segments   : dict,
    feat_data,
    feature_set,
    y_train,
    horizon         : int   = 1,
    storm_thr       : float = -50.0,
    dst_col         : str   = 'dst',
) -> dict:
    """
    Evaluate each model in models_dict on every segment in eval_segments,
    at a single fixed horizon. Pure computation — no MLflow side effects,
    no printing.

    Parameters
    ----------
    models_dict   : dict — {model_name: fitted model}
    eval_segments : dict — {seg_name: boolean mask into feat_data}
    feat_data     : DataFrame — full feature matrix
    feature_set   : list of str — feature columns to select from feat_data
    y_train       : array-like — training series for MASE denominator
    horizon       : int — forecast horizon in hours (default 1)
    storm_thr     : float — storm threshold in nT (default -50)
    dst_col       : str — column name for current Dst (default 'dst')

    Returns
    -------
    dict — {model_name: {seg_name: metrics_dict}}
    """
    target_col = f'dst_target_{horizon}h'
    results = {}
    for model_name, model in models_dict.items():
        results[model_name] = {}
        for seg_name, seg_mask in eval_segments.items():
            y_pred = model.predict(feat_data.loc[seg_mask, feature_set])
            results[model_name][seg_name] = compute_metrics(
                y_true    = feat_data.loc[seg_mask, target_col],
                y_pred    = y_pred,
                y_train   = y_train,
                y_persist = feat_data.loc[seg_mask, dst_col].values,
                storm_thr = storm_thr,
                horizon   = horizon,
            )
    return results


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_horizon(
    model_name    : str,
    seg_name      : str,
    h             : int,
    model,
    X_seg         : pd.DataFrame,
    y_true,
    y_train,
    y_persist      = None,
    storm_thr     : float = -50.0,
    extra_params  : dict  = None,
) -> dict:
    """
    Evaluate model at one horizon and log a nested MLflow child run.

    Parameters
    ----------
    model_name   : str — model identifier used in run name and tags
    seg_name     : str — segment name (e.g. 'val_main', 'val_storm')
    h            : int — forecast horizon in hours
    model        : fitted sklearn estimator or Pipeline
    X_seg        : DataFrame — evaluation features
    y_true       : array-like — observed Dst(t+h)
    y_train      : array-like — Train_1 Dst for MASE denominator
    y_persist    : array-like or None — persistence predictions for DM test
    storm_thr    : float — storm threshold in nT (default -50)
    extra_params : dict or None — additional params to log (e.g. alpha, beta)

    Returns
    -------
    dict — metrics from compute_metrics()
    """
    y_pred  = model.predict(X_seg)
    metrics = compute_metrics(
        y_true    = y_true,
        y_pred    = y_pred,
        y_train   = y_train,
        y_persist = y_persist,
        storm_thr = storm_thr,
        horizon   = h,
    )

    params = {'horizon': h, 'storm_thr': storm_thr}
    if extra_params:
        params.update(extra_params)

    log_metrics_run(
        run_name = f'{model_name}_{seg_name}_h{h}',
        tags     = {'model_type': model_name, 'evaluation_set': seg_name},
        params   = params,
        metrics  = metrics,
        nested   = True,
    )

    extra_display = {k: v for k, v in (extra_params or {}).items()
                     if isinstance(v, float)}
    print_horizon_metrics(h, metrics, extra=extra_display or None)

    return metrics


def run_segment(
    model_name      : str,
    seg_name        : str,
    models          : dict,
    X_seg_fn,
    y_true_fn,
    y_train,
    y_persist_fn     = None,
    storm_thr       : float = -50.0,
    k_horizons      : list  = None,
    extra_params_fn  = None,
) -> dict:
    """
    Evaluate model for all horizons in one segment.
    Opens a parent MLflow run and delegates to run_horizon().

    Parameters
    ----------
    model_name      : str — model identifier
    seg_name        : str — segment name
    models          : dict — {horizon: fitted model}
    X_seg_fn        : callable — f(h) -> DataFrame of features for segment
    y_true_fn       : callable — f(h) -> Series of targets for segment
    y_train         : array-like — Train_1 Dst for MASE denominator
    y_persist_fn    : callable or None — f(h) -> array of persistence predictions.
                      Pass None to skip DM test for all horizons.
    storm_thr       : float — storm threshold in nT (default -50)
    k_horizons      : list — forecast horizons (default [1,3,7,12,21])
    extra_params_fn : callable or None — f(h, model) -> dict of extra params to log

    Returns
    -------
    dict — {horizon: metrics}
    """
    if k_horizons is None:
        k_horizons = [1, 3, 7, 12, 21]

    print(f'\n── Segment: {seg_name} ──────────────────────────────────────')

    with safe_mlflow_run(run_name=f'{model_name}_{seg_name}') as run:
        if run is not None:
            mlflow.set_tag('model_type',     model_name)
            mlflow.set_tag('evaluation_set', seg_name)

        return {
            h: run_horizon(
                model_name   = model_name,
                seg_name     = seg_name,
                h            = h,
                model        = models[h],
                X_seg        = X_seg_fn(h),
                y_true       = y_true_fn(h),
                y_train      = y_train,
                y_persist    = y_persist_fn(h) if y_persist_fn else None,
                storm_thr    = storm_thr,
                extra_params = extra_params_fn(h, models[h]) if extra_params_fn else None,
            )
            for h in k_horizons
        }
