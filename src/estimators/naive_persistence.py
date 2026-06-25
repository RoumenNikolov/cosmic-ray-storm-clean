import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

"""
Naive persistence baseline estimator.

The simplest possible forecast: the geomagnetic field h hours from now
will be identical to its current value:

    ŷ(t+h) = D_st(t)

No learning occurs — fit() is a no-op. Serves as the absolute lower bound
for all subsequent models.
"""


class NaivePersistence(BaseEstimator, RegressorMixin):
    """
    Naive persistence baseline — predicts current Dst as future Dst.

    Parameters
    ----------
    dst_col : str — column name for current Dst (default 'dst')
    """

    def __init__(self, dst_col: str = 'dst'):
        self.dst_col = dst_col

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.dst_col].values
        raise ValueError(f"X must be a DataFrame with column '{self.dst_col}'")