# src/config.py
"""
Project-wide constants for the cosmic ray storm prediction project.
"""

STORM_THR       : float = -50.0              # nT — storm threshold [BUR75]
K_HORIZONS      : list  = [1, 3, 7, 12, 21] # forecasting horizons in hours
EXPERIMENT_NAME : str   = 'cosmic_ray_storm_prediction'
