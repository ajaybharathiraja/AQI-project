"""Inference on Phase 6 engineered rows, not raw sensor observations.

Usage: python -m ML.predict --model ML/saved_models/random_forest.joblib
       --input features.csv --output predictions.csv --trusted-model
Joblib can execute code: only load artifacts whose origin you trust.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def checked_features(frame, names):
    """Select exact training order; discard metadata, reject missing/nonfinite X."""
    if len(set(names)) != len(names) or frame.columns.duplicated().any():
        raise ValueError('Duplicate feature names')
    absent = [name for name in names if name not in frame]
    if absent:
        raise ValueError(f'Missing engineered features: {absent}')
    try:
        result = frame[names].astype('float32')
    except (TypeError, ValueError) as error:
        raise ValueError('Features must be numeric') from error
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise ValueError('Features must be nonempty and finite; use saved preprocessing state')
    return result


def predict_frame(model_path, frame, trusted=False):
    """Return nonnegative pollutant predictions, preserving input rows.

    All models use the same predeclared zero-floor physical constraint. Neither
    input labels nor timestamps enter X; no fitting or temporal splitting occurs.
    """
    if not trusted:
        raise ValueError('Explicit trust required before loading executable joblib artifact')
    import joblib
    model_path = Path(model_path)
    metadata = json.loads(model_path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != metadata['model_sha256']:
        raise ValueError('Model checksum mismatch')
    X = checked_features(frame, metadata['features'])
    model = joblib.load(model_path)
    prediction = np.asarray(model.predict(X), dtype=float)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid model predictions')
    return np.maximum(0, prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-model', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if destination == args.input.resolve() or destination.exists():
        raise ValueError('Choose a new output file; input cannot be overwritten')
    for name in ('DATA', 'PROCESSED_DATA', 'FEATURE_ENGINEERING'):
        if root / name in destination.parents:
            raise ValueError('Prediction output cannot overwrite source directories')
    frame = pd.read_csv(args.input)
    result = frame[[c for c in ('forecast_origin', 'target_start', 'target_end', 'district', 'port') if c in frame]].copy()
    result['predicted_pm25_source_scale'] = predict_frame(args.model, frame, args.trusted_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
