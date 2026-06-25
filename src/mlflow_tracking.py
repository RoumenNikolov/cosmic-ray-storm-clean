# src/tracking.py
"""
MLflow configuration utilities for the cosmic ray storm prediction project.

Functions
---------
setup_mlflow(tracking_uri)
    Configure MLflow tracking URI and set the experiment.
    Must be called once at the top of every notebook before any mlflow.start_run() call.
"""

import os
import mlflow


EXPERIMENT_NAME = 'cosmic_ray_storm_prediction'


def setup_mlflow(tracking_uri: str = "./mlruns") -> mlflow.entities.Experiment:
    """
    Configure MLflow file store backend and set the experiment.

    Parameters
    ----------
    tracking_uri : str — path to mlruns directory relative to the notebook
                   (default '../mlruns' for notebooks/ subdirectory)

    Returns
    -------
    mlflow.entities.Experiment
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(EXPERIMENT_NAME)
