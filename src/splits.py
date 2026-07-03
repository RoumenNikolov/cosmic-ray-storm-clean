# src/splits.py
"""
Chronological split masks for the cosmic ray storm prediction project.

Defines the temporal boundaries, purge zones and boolean masks used
consistently across all notebooks (EDA, ML, HPO).

PURGE_H = 21 hours — equal to the longest forecasting horizon. Removing
PURGE_H hours from each side of a segment boundary prevents data leakage:
the target variable Dst(t+h) at hour t near a boundary would otherwise
depend on observations from the adjacent segment.

Segment structure (chronological):
    Train_1     : 1995-01-01 → 2003-10-14  (Solar Cycle 23 rising phase)
    Val_Storm   : 2003-10-15 → 2003-12-15  (Halloween 2003 superstorm)
    Train_2     : 2003-12-16 → 2008-12-31  (Solar Cycle 23 declining phase)
    Val_Main    : 2009-01-01 → 2014-12-31  (Solar Cycle 24 rising phase)
    Test_Active : 2015-01-01 → 2015-12-31  (Solar Cycle 24 maximum)
    Test_Quiet  : 2016-01-01 → 2023-07-10  (Solar Cycle 24-25 transition)

Note: the dates above are the raw BOUNDARIES values. build_masks() additionally
trims PURGE_H hours from the applicable start/end of each segment (per the
purge_start/purge_end flags below), so the effective date range used by each
mask is narrower than the nominal boundaries shown here — by up to 21 hours
at each purged edge.

train_mask = Train_1 | Train_2 (Val_Storm excluded despite date overlap).

Usage
-----
from src.splits import BOUNDARIES, PURGE_H, segment_mask, build_masks

masks = build_masks(feat['datetime'])
train_mask     = masks['train']
val_main_mask  = masks['val_main']
val_storm_mask = masks['val_storm']
"""

import pandas as pd


PURGE_H = 21

BOUNDARIES = {
    'train1_start':       pd.Timestamp('1995-01-01 00:00'),
    'train1_end':         pd.Timestamp('2003-10-14 23:00'),
    'val_storm_start':    pd.Timestamp('2003-10-15 00:00'),
    'val_storm_end':      pd.Timestamp('2003-12-15 23:00'),
    'train2_start':       pd.Timestamp('2003-12-16 00:00'),
    'train2_end':         pd.Timestamp('2008-12-31 23:00'),
    'val_main_start':     pd.Timestamp('2009-01-01 00:00'),
    'val_main_end':       pd.Timestamp('2014-12-31 23:00'),
    'test_active_start':  pd.Timestamp('2015-01-01 00:00'),
    'test_active_end':    pd.Timestamp('2015-12-31 23:00'),
    'test_quiet_start':   pd.Timestamp('2016-01-01 00:00'),
    'test_quiet_end':     pd.Timestamp('2023-07-10 23:00'),
}


def segment_mask(dt, start_key, end_key, purge_start=True, purge_end=True):
    """
    Boolean mask for a chronological segment with optional purge zones.

    Parameters
    ----------
    dt          : pd.Series — datetime column from feature matrix
    start_key   : str — key in BOUNDARIES for segment start
    end_key     : str — key in BOUNDARIES for segment end
    purge_start : bool — apply PURGE_H offset to start (default True)
    purge_end   : bool — apply PURGE_H offset to end (default True)

    Returns
    -------
    pd.Series of bool
    """
    start = BOUNDARIES[start_key]
    end   = BOUNDARIES[end_key]
    if purge_start:
        start = start + pd.Timedelta(hours=PURGE_H)
    if purge_end:
        end   = end   - pd.Timedelta(hours=PURGE_H)
    return (dt >= start) & (dt <= end)


def build_masks(data):
    """
    Build all segment masks for the project.

    Parameters
    ----------
    data : pd.Series — datetime column from feature matrix

    Returns
    -------
    dict of {str: pd.Series of bool}
    """
    return {
        'train'      : (
            segment_mask(data, 'train1_start', 'train1_end',
                         purge_start=False, purge_end=True) |
            segment_mask(data, 'train2_start', 'train2_end',
                         purge_start=True,  purge_end=True)
        ),
        'train1'     : segment_mask(data, 'train1_start', 'train1_end',
                                    purge_start=False, purge_end=True),
        'train2'     : segment_mask(data, 'train2_start', 'train2_end',
                                    purge_start=True,  purge_end=True),
        'val_storm'  : segment_mask(data, 'val_storm_start',   'val_storm_end'),
        'val_main'   : segment_mask(data, 'val_main_start',    'val_main_end'),
        'test_active': segment_mask(data, 'test_active_start', 'test_active_end'),
        'test_quiet' : segment_mask(data, 'test_quiet_start',  'test_quiet_end',
                                    purge_start=True, purge_end=False),
    }