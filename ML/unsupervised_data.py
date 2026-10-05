"""Validation shared by independent unsupervised components; no DS imports."""
import numpy as np
import pandas as pd

POLLUTANTS = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
IDENTITY = ['timestamp', 'district', 'port']


def validate_hourly(frame, pollutants):
    """Copy audited hourly interval-start observations, preserving original values.

    Missing measurements are allowed; invalid concentrations and duplicate keys
    are not. Input is the EDA complete-native-slot hourly view, not arbitrary raw
    CSV rows. Naive times retain the documented Asia/Kolkata assumption.
    """
    absent = set(IDENTITY + list(pollutants)) - set(frame.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    result = frame.copy()
    result['timestamp'] = pd.to_datetime(result.timestamp, errors='raise')
    if result.empty or result[IDENTITY].isna().any().any():
        raise ValueError('Nonempty data with complete identity is required')
    if not result.timestamp.eq(result.timestamp.dt.floor('h')).all():
        raise ValueError('Expected aligned hourly interval starts')
    if result.duplicated(IDENTITY).any():
        raise ValueError('Duplicate station/hour keys')
    for col in pollutants:
        values = pd.to_numeric(result[col], errors='raise')
        if (values.dropna() < 0).any() or not np.isfinite(values.dropna()).all():
            raise ValueError(f'Invalid {col} concentration')
        result[col] = values
    return result.sort_values(IDENTITY).reset_index(drop=True)
