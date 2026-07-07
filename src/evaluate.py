# src/evaluate.py
"""
Evaluation metrics for geomagnetic storm prediction.

All functions are pure — no fitting, no side effects, no visualisation.

Functions
---------
rmse(y_true, y_pred)
    Root Mean Squared Error.

r2(y_true, y_pred)
    R² coefficient of determination.

storm_rmse(y_true, y_pred, storm_thr)
    RMSE restricted to storm hours (y_true < storm_thr).

mase(y_true, y_pred, y_train)
    Mean Absolute Scaled Error [HYN06] — normalised by in-sample
    one-step persistence MAE on the training set.

diebold_mariano(y_true, y_pred_1, y_pred_2, h, criterion)
    Diebold-Mariano test [DM95] for equal predictive accuracy.
    Tests whether model 2 is significantly more accurate than model 1
    using sample variance of the loss differential series.

peak_timing_error(y_true, y_pred)
    Signed difference in hours between predicted and actual Dst minimum.
    Returns NaN when no storm is present in the evaluation window.

compute_metrics(y_true, y_pred, y_train, y_persist, storm_thr, horizon)
    Wrapper that returns all metrics as a dict in a single call.

References
----------
[HYN06] Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures
        of forecast accuracy. International Journal of Forecasting, 22(4),
        679-688.

[DM95]  Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive
        accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score as _r2_score


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_array(*args):
    """Convert all inputs to 1-D float64 numpy arrays."""
    return [np.asarray(a, dtype=np.float64).ravel() for a in args]


# ── Public API ────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like — observed values
    y_pred : array-like — predicted values

    Returns
    -------
    float
    """
    yt, yp = _to_array(y_true, y_pred)
    mask   = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def r2(y_true, y_pred) -> float:
    """
    R² coefficient of determination.

    Measures the proportion of variance in y_true explained by y_pred.
    A value of 1.0 indicates perfect prediction; 0.0 indicates the model
    performs no better than predicting the mean; negative values indicate
    the model performs worse than predicting the mean (i.e. worse than
    persistence at the same horizon for a stationary series).

    Parameters
    ----------
    y_true : array-like — observed values
    y_pred : array-like — predicted values

    Returns
    -------
    float — R² score, or NaN if no finite pairs exist
    """
    yt, yp = _to_array(y_true, y_pred)
    mask   = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return np.nan
    return float(_r2_score(yt[mask], yp[mask]))


def storm_rmse(y_true, y_pred, storm_thr: float = -50.0) -> float:
    """
    RMSE restricted to storm hours (y_true < storm_thr).

    A geomagnetic storm is defined as Dst < -50 nT following
    the convention used throughout this project.

    Parameters
    ----------
    y_true    : array-like — observed Dst values
    y_pred    : array-like — predicted Dst values
    storm_thr : float — storm threshold in nT (default -50)

    Returns
    -------
    float — storm RMSE, or NaN if no storm hours exist in the window
    """
    yt, yp = _to_array(y_true, y_pred)
    mask   = np.isfinite(yt) & np.isfinite(yp) & (yt < storm_thr)
    if mask.sum() == 0:
        warnings.warn(
            f"storm_rmse: no storm hours found (y_true < {storm_thr} nT). "
            "Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))

def neg_storm_rmse(y_true, y_pred, storm_thr: float = -50.0) -> float:
    """
    Negative Storm RMSE for use with sklearn's make_scorer.

    Returns the negative of storm_rmse() so that RandomizedSearchCV,
    which maximises the scorer, selects the configuration with the
    lowest Storm RMSE.

    Parameters
    ----------
    y_true    : array-like — observed Dst values
    y_pred    : array-like — predicted Dst values
    storm_thr : float — storm threshold in nT (default -50)

    Returns
    -------
    float — negative Storm RMSE
    """
    return -storm_rmse(y_true, y_pred, storm_thr=storm_thr)


def mase(y_true, y_pred, y_train) -> float:
    """
    Mean Absolute Scaled Error [HYN06].

    Normalises MAE by the in-sample mean absolute one-step difference
    on the training set. A value below 1.0 indicates the model
    outperforms naive one-step persistence on the training data.

    Formula
    -------
    MASE = MAE(y_true, y_pred) / (1/(T-1) * sum(|y_train[t] - y_train[t-1]|))

    The denominator is independent of the forecast horizon — it is the
    same scaling constant regardless of h.

    Parameters
    ----------
    y_true  : array-like — observed values (evaluation set)
    y_pred  : array-like — predicted values (evaluation set)
    y_train : array-like — training set Dst series for denominator

    Returns
    -------
    float
    """
    yt, yp, ytr = _to_array(y_true, y_pred, y_train)

    eval_mask = np.isfinite(yt) & np.isfinite(yp)
    if eval_mask.sum() == 0:
        return np.nan

    mae_num = np.mean(np.abs(yt[eval_mask] - yp[eval_mask]))

    # Diff is computed on the original (ordered) sequence first; only then
    # are non-finite differences dropped. This avoids silently bridging two
    # non-adjacent timestamps across a gap/NaN in y_train — if ytr[i] is NaN,
    # both diff[i-1] and diff[i] become NaN and are excluded, rather than
    # diff(ytr) implicitly treating the surviving neighbours as adjacent.
    train_diffs = np.abs(np.diff(ytr))
    train_diffs = train_diffs[np.isfinite(train_diffs)]
    if len(train_diffs) == 0:
        return np.nan

    denominator = np.mean(train_diffs)
    if denominator == 0.0:
        warnings.warn(
            "mase: denominator is zero (constant training series). "
            "Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    return float(mae_num / denominator)


def diebold_mariano(
    y_true,
    y_pred_1,
    y_pred_2,
    h        : int = 1,
    criterion: str = "squared",
    ) -> tuple:
    """
    Diebold-Mariano test for equal predictive accuracy [DM95].

    Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) is the
    loss differential. A positive DM statistic indicates that model 2
    is more accurate than model 1.

    Variance is estimated as the sample variance of the loss differential
    series, divided by the sample size. This is the standard formulation
    from Diebold & Mariano (1995). At large sample sizes (n > 50,000),
    even small systematic differences produce significant test statistics —
    results should be interpreted alongside practical metrics such as
    Storm RMSE and feature importance.

    Parameters
    ----------
    y_true   : array-like — observed values
    y_pred_1 : array-like — predictions from model 1 (baseline)
    y_pred_2 : array-like — predictions from model 2 (candidate)
    h        : int — forecast horizon (kept for API compatibility)
    criterion: str — loss function: 'squared' (default) or 'absolute'

    Returns
    -------
    (dm_stat, p_value) : tuple[float, float]
    """
    yt, yp1, yp2 = _to_array(y_true, y_pred_1, y_pred_2)
    mask = np.isfinite(yt) & np.isfinite(yp1) & np.isfinite(yp2)

    if mask.sum() < 10:
        return np.nan, np.nan

    e1 = yt[mask] - yp1[mask]
    e2 = yt[mask] - yp2[mask]

    if criterion == "squared":
        d = e1 ** 2 - e2 ** 2
    elif criterion == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"criterion must be 'squared' or 'absolute', got '{criterion}'")

    dm_stat = d.mean() / (d.std() / np.sqrt(len(d)))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    return float(dm_stat), p_value


def peak_timing_error(y_true, y_pred, storm_thr: float = -50.0) -> float:
    """
    Signed peak timing error in hours.

    Computes the difference between the index of the predicted Dst minimum
    and the index of the observed Dst minimum within the evaluation window.

    A positive value means the model predicts the storm peak too late;
    a negative value means too early.

    Returns NaN when no storm (y_true < storm_thr) is present, or when
    the evaluation window contains fewer than 2 finite observations.

    Parameters
    ----------
    y_true    : array-like — observed Dst values
    y_pred    : array-like — predicted Dst values
    storm_thr : float — storm threshold in nT (default -50.0)

    Returns
    -------
    float — signed timing error in hours, or NaN
    """
    yt, yp = _to_array(y_true, y_pred)
    mask   = np.isfinite(yt) & np.isfinite(yp)

    if mask.sum() < 2:
        return np.nan

    if not np.any(yt[mask] < storm_thr):
        return np.nan

    idx_true = int(np.argmin(yt[mask]))
    idx_pred = int(np.argmin(yp[mask]))

    return float(idx_pred - idx_true)


# ── Master wrapper ────────────────────────────────────────────────────────────

def compute_metrics(
    y_true,
    y_pred,
    y_train,
    y_persist  = None,
    storm_thr  : float = -50.0,
    horizon    : int   = 1,
) -> dict:
    """
    Compute all evaluation metrics in a single call.

    Parameters
    ----------
    y_true    : array-like — observed Dst values (evaluation set)
    y_pred    : array-like — predicted Dst values (candidate model)
    y_train   : array-like — training set Dst series (for MASE denominator)
    y_persist : array-like or None — persistence predictions for DM test baseline.
                If None, the DM test is skipped and dm_stat/dm_pvalue are NaN.
                Pass None for the Naive Persistence baseline where the DM test
                is undefined (model compared against itself).
    storm_thr : float — storm threshold in nT (default -50)
    horizon   : int — forecast horizon in hours (kept for API compatibility
                with diebold_mariano(); not used in the variance calculation)

    Returns
    -------
    dict with keys:
        rmse            — overall RMSE
        r2              — R² coefficient of determination
        storm_rmse      — RMSE on storm hours only
        mase            — Mean Absolute Scaled Error vs one-step persistence
        dm_stat         — Diebold-Mariano statistic (positive = candidate
                           model [y_pred] more accurate than y_persist)
        dm_pvalue       — DM two-sided p-value
        peak_timing_err — signed timing error in hours (NaN if no storm)
        n_eval          — number of finite evaluation pairs
        n_storm         — number of storm hours in evaluation set
    """
    yt, yp, ytr = _to_array(y_true, y_pred, y_train)
    finite_mask  = np.isfinite(yt) & np.isfinite(yp)
    storm_mask   = finite_mask & (yt < storm_thr)

    if y_persist is not None:
        ypers = _to_array(y_persist)[0]
        dm_stat, dm_pvalue = diebold_mariano(
            yt, ypers, yp, h=horizon, criterion="squared"
        )
    else:
        dm_stat, dm_pvalue = np.nan, np.nan

    return {
        "rmse"           : rmse(yt, yp),
        "r2"             : r2(yt, yp),
        "storm_rmse"     : storm_rmse(yt, yp, storm_thr=storm_thr),
        "mase"           : mase(yt, yp, ytr),
        "dm_stat"        : dm_stat,
        "dm_pvalue"      : dm_pvalue,
        "peak_timing_err": peak_timing_error(yt, yp, storm_thr=storm_thr),
        "n_eval"         : int(finite_mask.sum()),
        "n_storm"        : int(storm_mask.sum()),
    }

    


def dm_hybrid(true_vals, preds_a, preds_b, h=7):
    """
    Two-sided Diebold-Mariano test — thin convenience wrapper around diebold_mariano().

    Positive statistic means preds_b is more accurate than preds_a.

    Note: the test is two-sided, matching diebold_mariano()'s own p-value formula
    (2 * (1 - Phi(|stat|))). An earlier version of this notebook labelled one of two
    duplicated copies of this function "one-sided" — both copies called the same
    underlying two-sided computation; this docstring reflects what is actually computed.

    Parameters
    ----------
    true_vals : array-like — observed values
    preds_a   : array-like — predictions from model A (baseline)
    preds_b   : array-like — predictions from model B (candidate)
    h         : int — forecast horizon (default 7)

    Returns
    -------
    (stat, p) : tuple[float, float]
    """
    stat, p = diebold_mariano(true_vals, preds_a, preds_b, h=h, criterion="squared")
    return stat, p
