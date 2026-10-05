"""Persistence baseline independent of custom data structures and ML libraries."""
import numpy as np


class PersistenceRegressor:
    """Predict last eligible PM2.5 hour, or training-target median if unavailable.

    Phase 6 lag 1 ends an hour BEFORE origin; the interval ending at origin is
    intentionally excluded. Fit learns only a fallback. Predict is O(n).
    """
    def __init__(self, target='pm25'):
        self.target = target

    def fit(self, X, y):
        values = np.asarray(y, dtype=float)
        if not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Training targets must be finite, nonnegative and nonempty')
        if len(X) != len(values):
            raise ValueError('Feature and target lengths differ')
        self.fallback_ = float(np.median(values))
        return self

    def predict(self, X):
        if not hasattr(self, 'fallback_'):
            raise ValueError('Baseline must be fitted first')
        col = f'{self.target}_lag_1h'
        missing = f'{col}__missing'
        if col not in X or missing not in X:
            raise ValueError('Persistence requires lag 1 and its missingness indicator')
        values = X[col].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values >= 0) & X[missing].eq(0).to_numpy()
        return np.where(valid, values, self.fallback_)

    def get_params(self, deep=True):
        return {'target': self.target}
