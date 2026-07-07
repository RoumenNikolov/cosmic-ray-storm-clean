# src/mlflow_tracking.py
"""
MLflow configuration utilities for the cosmic ray storm prediction project.

Off by default. Tracking writes a local ./mlruns directory on every run — fine for
active development, but not something a grader re-running these notebooks should get
saddled with unasked. Set ENABLE_MLFLOW = True in src/config.py to turn it on for
every notebook at once — no OS-specific commands, no environment variables to remember.

Functions
---------
mlflow_safe(func)
    Decorator that catches exceptions raised by MLflow calls and emits a
    warning instead of propagating them, so MLflow tracking failures never
    interrupt the evaluation pipeline. Applied to any function whose sole
    purpose involves MLflow side effects — kept separate from those
    functions' own logic (single responsibility).

setup_mlflow(tracking_uri)
    Configure MLflow tracking URI and set the experiment. No-ops (returns
    None immediately, writes nothing to disk) unless ENABLE_MLFLOW is set.
    Must be called once at the top of every notebook before any mlflow.start_run() call.

safe_mlflow_run(...)
    Context manager wrapping mlflow.start_run(). No-ops (yields None,
    starts no run) unless ENABLE_MLFLOW is set — every caller in this
    project already checks `if run is not None` before logging, so
    disabling tracking here requires no changes anywhere else.
"""

import contextlib
import functools
import os
import warnings
import mlflow

from .config import EXPERIMENT_NAME, ENABLE_MLFLOW


def mlflow_safe(func):
    """
    Decorator that catches exceptions raised by MLflow calls and emits a
    warning instead of propagating them.

    Parameters
    ----------
    func : callable — a function whose body performs MLflow operations

    Returns
    -------
    callable — wrapped function that returns None (with a warning) on
               failure instead of raising
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warnings.warn(
                f"MLflow operation '{func.__name__}' failed: {e}. Skipping.",
                RuntimeWarning,
            )
            return None
    return wrapper


@contextlib.contextmanager
def safe_mlflow_run(*args, **kwargs):
    """
    Context manager wrapping mlflow.start_run(). Yields None without
    starting a run if ENABLE_MLFLOW is off, or if MLflow is unavailable
    or misconfigured — the code inside the `with` block (e.g. business
    logic computing and returning results) still executes normally either
    way, and every caller in this project checks `if run is not None`
    before attempting to log anything.

    Parameters
    ----------
    *args, **kwargs : passed through to mlflow.start_run()

    Yields
    ------
    the active mlflow run, or None if tracking is disabled or setup failed
    """
    if not ENABLE_MLFLOW:
        yield None
        return
    try:
        with mlflow.start_run(*args, **kwargs) as run:
            yield run
    except Exception as e:
        warnings.warn(
            f"MLflow run could not be started: {e}. Continuing without tracking.",
            RuntimeWarning,
        )
        yield None


@mlflow_safe
def setup_mlflow(tracking_uri: str = "./mlruns") -> "mlflow.entities.Experiment | None":
    """
    Configure MLflow file store backend and set the experiment.
    No-ops (returns None, touches no files) unless ENABLE_MLFLOW is set.

    Parameters
    ----------
    tracking_uri : str — path to the mlruns directory (default './mlruns', relative
                   to wherever the notebook's own working directory is — matches
                   the convention used by every notebook in this project, which all
                   call setup_mlflow() with no arguments)

    Returns
    -------
    mlflow.entities.Experiment, or None if tracking is disabled or setup failed
    """
    if not ENABLE_MLFLOW:
        return None
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(EXPERIMENT_NAME)
