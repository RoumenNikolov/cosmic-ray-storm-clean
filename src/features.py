# src/features.py
"""
Feature set definitions for the cosmic ray storm prediction project.

Constructs the four predictor configurations (MODEL_A–D) used in the
ablation study and operational forecasting model.

Usage
-----
from src.features import build_feature_sets

feature_sets = build_feature_sets(selected_features)
MODEL_C_OMNI_DNEUTRON = feature_sets['MODEL_C_OMNI_DNEUTRON']
"""

NEUTRON_RAW     = ['neutron_counts']
NEUTRON_DERIVED = ['d_neutron']
NEUTRON_HISTORY = ['neutron_counts_lag3', 'neutron_counts_lag7']
NEUTRON_ALL     = NEUTRON_RAW + NEUTRON_DERIVED + NEUTRON_HISTORY


def build_feature_sets(selected_features):
    """
    Build all predictor configurations from the selected feature list.

    Parameters
    ----------
    selected_features : list of str — output of feature selection

    Returns
    -------
    dict with keys:
        OMNI_FEATURES         — 20 OMNI solar wind features
        MODEL_A_OMNI          — OMNI only (baseline)
        MODEL_B_OMNI_RAW      — OMNI + neutron_counts
        MODEL_C_OMNI_DNEUTRON — OMNI + d_neutron
        MODEL_D_OMNI_HISTORY  — OMNI + d_neutron + lag3 + lag7
    """
    omni = [f for f in selected_features if f not in NEUTRON_ALL]
    return {
        'OMNI_FEATURES'         : omni,
        'MODEL_A_OMNI'          : omni,
        'MODEL_B_OMNI_RAW'      : omni + NEUTRON_RAW,
        'MODEL_C_OMNI_DNEUTRON' : omni + NEUTRON_DERIVED,
        'MODEL_D_OMNI_HISTORY'  : omni + NEUTRON_DERIVED + NEUTRON_HISTORY,
    }