"""Exact saved-feature contract and checksum-verified SVM inference. Never fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ML.predict import checked_features
from ML.model_evaluation import sha256


def validate_features(frame, names):
    if frame.columns.duplicated().any() or set(frame.columns) != set(names):
        raise ValueError(f'Exact features required; missing={sorted(set(names)-set(frame.columns))}; unexpected={sorted(set(frame.columns)-set(names))}')
    for name in names:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f'{name} must contain actual numbers, not strings/booleans')
    X = checked_features(frame, names)
    binary = [n for n in names if n.endswith('__missing') or n.startswith(('district_', 'station_'))]
    if binary and not X[binary].isin([0, 1]).all().all():
        raise ValueError('Location and missingness encodings must be binary')
    for prefix in ('district_', 'station_'):
        columns = [n for n in names if n.startswith(prefix)]
        if columns and not X[columns].sum(axis=1).eq(1).all():
            raise ValueError(f'Exactly one {prefix} location bucket must be active')
    for name in names:
        if name.endswith('__missing'):
            continue
        if name.startswith(tuple(p + '_' for p in ('pm25', 'pm10', 'no', 'no2', 'nh3', 'so2', 'co', 'o3'))) and (X[name] < 0).any():
            raise ValueError(f'Negative pollutant feature: {name}')
    return X


class SVMForecaster:
    def __init__(self, directory, *, trusted=False):
        if not trusted:
            raise ValueError('Explicit trust is required for executable local joblib artifacts')
        import joblib
        directory = Path(directory)
        self.metadata = json.loads((directory / 'svm_metadata.json').read_text(encoding='utf-8'))
        path = directory / 'svm_model.joblib'
        if sha256(path) != self.metadata['model_sha256']:
            raise ValueError('SVM model checksum mismatch')
        self.pipeline = joblib.load(path)
        if list(self.pipeline.feature_names_in_) != self.metadata['features']:
            raise ValueError('SVM feature metadata/pipeline mismatch')
        if set(self.pipeline.classes_) != set(self.metadata['learned_classes']):
            raise ValueError('SVM learned class metadata mismatch')

    def predict(self, frame):
        X = validate_features(frame, self.metadata['features'])
        prediction = self.pipeline.predict(X)
        if not set(prediction).issubset(self.metadata['target_classes']):
            raise ValueError('Model returned an unknown category')
        return prediction.tolist()
