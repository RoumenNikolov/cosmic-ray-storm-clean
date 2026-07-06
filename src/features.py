# src/features.py
"""
Feature set definitions for the cosmic ray storm prediction project.

Constructs the four predictor configurations (MODEL_A–D) used in the
ablation study and operational forecasting model.

NEUTRON_RAW and NEUTRON_DERIVED are fixed column names (the raw neutron
count and its first difference are always single, specific columns).

NEUTRON_HISTORY is NOT a fixed constant — which `neutron_counts_lag*`
features survive feature selection depends on the output of the
selection pipeline (e.g. lag3+lag12 in one run, lag3+lag7 in another).
It is therefore derived at call time from `selected_features` via
get_neutron_history() / get_neutron_all(), rather than hardcoded — a
previous hardcoded version of this constant (['neutron_counts_lag3',
'neutron_counts_lag7']) drifted out of sync with the actual selected
features once feature selection was rerun.

Usage
-----
from src.features import build_feature_sets

feature_sets = build_feature_sets(selected_features)
MODEL_C_OMNI_DNEUTRON = feature_sets['MODEL_C_OMNI_DNEUTRON']
"""

NEUTRON_RAW     = ['neutron_counts']
NEUTRON_DERIVED = ['d_neutron']

NEUTRON_LAG_PREFIX = 'neutron_counts_lag'


def get_neutron_history(selected_features):
    """
    Return the neutron_counts_lag* features actually present in
    `selected_features`, sorted alphabetically.

    Parameters
    ----------
    selected_features : list of str — output of feature selection

    Returns
    -------
    list of str — surviving neutron_counts_lag* feature names
    """
    return sorted(
        f for f in selected_features if f.startswith(NEUTRON_LAG_PREFIX)
    )


def get_neutron_all(selected_features):
    """
    Return the full neutron_counts family (raw + derived + surviving
    lag features) actually present given `selected_features`.

    Parameters
    ----------
    selected_features : list of str — output of feature selection

    Returns
    -------
    list of str
    """
    return NEUTRON_RAW + NEUTRON_DERIVED + get_neutron_history(selected_features)


def build_feature_sets(selected_features):
    """
    Build all predictor configurations from the selected feature list.

    Parameters
    ----------
    selected_features : list of str — output of feature selection

    Returns
    -------
    dict with keys:
        OMNI_FEATURES         — OMNI solar wind features (count depends on
                                 how many neutron/derived features survive
                                 selection; not a fixed number)
        MODEL_A_OMNI          — OMNI only (baseline)
        MODEL_B_OMNI_RAW      — OMNI + neutron_counts
        MODEL_C_OMNI_DNEUTRON — OMNI + d_neutron
        MODEL_D_OMNI_HISTORY  — OMNI + d_neutron + whichever
                                 neutron_counts_lag* features survived
                                 selection (see get_neutron_history())
    """
    neutron_history = get_neutron_history(selected_features)
    neutron_all      = get_neutron_all(selected_features)
    omni = [f for f in selected_features if f not in neutron_all]
    return {
        'OMNI_FEATURES'         : omni,
        'MODEL_A_OMNI'          : omni,
        'MODEL_B_OMNI_RAW'      : omni + NEUTRON_RAW,
        'MODEL_C_OMNI_DNEUTRON' : omni + NEUTRON_DERIVED,
        'MODEL_D_OMNI_HISTORY'  : omni + NEUTRON_DERIVED + neutron_history,
    }
