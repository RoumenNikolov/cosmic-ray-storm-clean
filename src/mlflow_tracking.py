# src/tracking.py
"""
MLflow configuration utilities for the cosmic ray storm prediction project.

Functions
---------
mlflow_safe(func)
    Decorator that catches exceptions raised by MLflow calls and emits a
    warning instead of propagating them, so MLflow tracking failures never
    interrupt the evaluation pipeline. Applied to any function whose sole
    purpose involves MLflow side effects — kept separate from those
    functions' own logic (single responsibility).

setup_mlflow(tracking_uri)
    Configure MLflow tracking URI and set the experiment.
    Must be called once at the top of every notebook before any mlflow.start_run() call.
"""

import contextlib
import functools
import os
import warnings
import mlflow


EXPERIMENT_NAME = 'cosmic_ray_storm_prediction'


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
    Context manager wrapping mlflow.start_run(). Falls back to a no-op
    context (yielding None) if MLflow is unavailable or misconfigured,
    instead of raising and interrupting the caller's own return value.

    Unlike mlflow_safe, this only guards the MLflow context itself — the
    code inside the `with` block (e.g. business logic computing and
    returning results) still executes normally even if entering the
    MLflow run fails.

    Parameters
    ----------
    *args, **kwargs : passed through to mlflow.start_run()

    Yields
    ------
    the active mlflow run, or None if MLflow setup failed
    """
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

    Parameters
    ----------
    tracking_uri : str — path to mlruns directory relative to the notebook
                   (default '../mlruns' for notebooks/ subdirectory)

    Returns
    -------
    mlflow.entities.Experiment, or None if MLflow setup failed
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(EXPERIMENT_NAME)
