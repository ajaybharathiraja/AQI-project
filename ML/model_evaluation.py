"""Phase 7 runner: train -> validation selection lock -> one final test pass.

Run: python -m ML.model_evaluation. Dependencies: ML/requirements.txt.
The default output is single-use; a completed/locked experiment is not silently
rerun against the same test set. No DATA_STRUCTURES imports or model tuning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import pandas as pd
from EDA.common import fingerprint
from FEATURE_ENGINEERING.build_features import temporal_split
from ML.baseline_model import PersistenceRegressor
from ML.predict import checked_features, predict_frame
from ML.random_forest import create_model as create_forest
from ML.xgboost_model import create_model as create_xgboost


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def safe(item):
        # XGBoost's `missing` parameter is NaN. JSON records it symbolically;
        # trained estimators are restored from joblib, not rebuilt from JSON.
        if isinstance(item, float) and not np.isfinite(item):
            return str(item)
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        return item
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding='utf-8')


def regression_metrics(actual, predicted):
    """MAE, RMSE, R². Undefined R² (constant target / <2 rows) is None."""
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metric inputs must be matching nonempty finite vectors')
    error = y - p
    sse = float(np.dot(error, error))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {'MAE': float(np.abs(error).mean()), 'RMSE': float(np.sqrt(np.mean(error ** 2))),
            'R2': 1 - sse / sst if len(y) > 1 and sst > 0 else None}


def select_model(validation_rows):
    """Lowest validation MAE; ties by RMSE then model name. Test rows forbidden."""
    if not validation_rows or any(row['split'] != 'validation' for row in validation_rows):
        raise ValueError('Selection accepts validation results only')
    if any(not np.isfinite(row['MAE']) or not np.isfinite(row['RMSE']) for row in validation_rows):
        raise ValueError('Nonfinite validation scores')
    return min(validation_rows, key=lambda row: (row['MAE'], row['RMSE'], row['model']))['model']


def read_partitions(path, names, manifest):
    """Read requested partitions only into memory; validate chronology and labels."""
    parts = []
    for chunk in pd.read_csv(path, chunksize=20000,
                             parse_dates=['forecast_origin', 'target_start', 'target_end']):
        part = chunk.loc[chunk.split.isin(names)].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError('Requested partitions are empty')
    frame = pd.concat(parts, ignore_index=True)
    if set(frame.split) != set(names) or not frame.forecast_origin.is_monotonic_increasing:
        raise ValueError('Missing partition or unsorted time-series dataset')
    expected = temporal_split(frame, manifest['validation_start'], manifest['test_start'])
    if not frame.split.eq(expected).all():
        raise ValueError('Temporal split/label boundary violation')
    if frame.duplicated(['district', 'port', 'forecast_origin']).any():
        raise ValueError('Duplicate station/origin examples')
    if not np.isfinite(frame.target_value).all() or frame.target_value.lt(0).any():
        raise ValueError('Invalid target values')
    if not (frame.target_end - frame.forecast_origin).eq(pd.Timedelta(hours=1)).all() or not frame.target_start.eq(frame.forecast_origin).all():
        raise ValueError('This experiment requires the one-hour Phase 6 contract')
    return frame


def range_metadata(frame):
    return {'rows': len(frame), 'first_origin': str(frame.forecast_origin.min()),
            'last_origin': str(frame.forecast_origin.max()), 'last_target_end': str(frame.target_end.max()),
            'stations': len(frame[['district', 'port']].drop_duplicates())}


def score(model, X, y, name, split, training_seconds):
    start = perf_counter()
    raw = np.asarray(model.predict(X), dtype=float)
    prediction = np.maximum(0, raw)
    elapsed = perf_counter() - start
    values = regression_metrics(y, prediction)
    row = {'model': name, 'split': split, 'rows': len(y), **values,
           'training_seconds': training_seconds, 'prediction_seconds': elapsed,
           'prediction_ms_per_row': elapsed * 1000 / len(y),
           'negative_predictions_clipped': int((raw < 0).sum())}
    return row, prediction


def run(root, n_jobs=2):
    """Execute the prespecified three-candidate experiment once, saving artifacts."""
    import joblib
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')
    root = Path(root).resolve()
    feature_dir = root / 'REPORTS' / 'FEATURE_ENGINEERING'
    output, models_dir = root / 'REPORTS' / 'ML', root / 'ML' / 'saved_models'
    if (output / 'selection_lock.json').exists() or any(models_dir.glob('*.joblib')):
        raise ValueError('An experiment is already saved/locked. Do not repeatedly evaluate the test set; review the existing report.')
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads((feature_dir / 'preprocessing_state.json').read_text())
    manifest = json.loads((feature_dir / 'run_manifest.json').read_text())
    if state['target'] != 'pm25' or state['horizon_hours'] != 1:
        raise ValueError('This prespecified experiment targets one-hour PM2.5')
    dataset = feature_dir / 'ml_ready_dataset.csv.gz'
    if sha256(dataset) != manifest['dataset_sha256']:
        raise ValueError('Feature dataset checksum mismatch')
    before = fingerprint(root)
    if before != manifest['source_after']:
        raise ValueError('Source data changed since feature engineering; audit before training')
    features = state['feature_columns']
    forbidden = {'target_value', 'split', 'forecast_origin', 'target_start', 'target_end', 'district', 'port'}
    if forbidden.intersection(features):
        raise ValueError('Metadata/label leakage in feature allowlist')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy', 'joblib', 'threadpoolctl', 'cloudpickle', 'narwhals')}
    contract = {'TARGET': 'Hourly PM2.5 mean in original numeric scale; physical units unverified',
                'FEATURES': features, 'FORECAST_HORIZON': 'At t predict [t,t+1h); every measured feature ends <t',
                'selection': 'Validation MAE, then RMSE, then model name',
                'postprocessing': 'Clip negative predictions to zero for all candidates',
                'candidate_order': ['persistence', 'random_forest', 'xgboost'],
                'test_policy': 'Freeze all models and selection before one final evaluation; no refit',
                'dataset_sha256': manifest['dataset_sha256'], 'preprocessing_sha256': sha256(feature_dir / 'preprocessing_state.json'),
                'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'experiment_protocol.json', contract)
    development = read_partitions(dataset, ['train', 'validation'], manifest)
    train = development.loc[development.split.eq('train')]
    validation = development.loc[development.split.eq('validation')]
    X_train, X_val = checked_features(train, features), checked_features(validation, features)
    y_train, y_val = train.target_value.to_numpy(), validation.target_value.to_numpy()
    ranges = {'train': range_metadata(train), 'validation': range_metadata(validation)}
    candidates = [('persistence', PersistenceRegressor()), ('random_forest', create_forest(n_jobs)), ('xgboost', create_xgboost(n_jobs))]
    rows, records, fitted = [], {}, {}
    for name, model in candidates:
        print(f'Training {name} on {len(train):,} chronological training rows', flush=True)
        trained_at = datetime.now(timezone.utc).isoformat()
        start = perf_counter()
        model.fit(X_train, y_train)
        training_seconds = perf_counter() - start
        row, predictions = score(model, X_val, y_val, name, 'validation', training_seconds)
        rows.append(row)
        fitted[name] = model
        path = models_dir / f'{name}.joblib'
        joblib.dump(model, path, compress=3)
        record = {'training_date_utc': trained_at, 'algorithm': type(model).__name__,
                  'parameters': model.get_params(), 'features': features, 'target': 'pm25',
                  'target_units': 'original numeric scale; units unverified', 'forecast_horizon_hours': 1,
                  'prediction_interval': '[forecast_origin, forecast_origin+1h)',
                  'measurement_cutoff': 'observation ends strictly before forecast_origin',
                  'dataset_range': dict(ranges), 'dataset_sha256': manifest['dataset_sha256'],
                  'preprocessing_state': state, 'preprocessing_sha256': contract['preprocessing_sha256'],
                  'algorithm_parameters_fixed_before_validation': True, 'random_seed': 42,
                  'postprocessing': 'max(0,prediction)', 'training_seconds': training_seconds,
                  'evaluation_results': {'validation': row}, 'versions': versions,
                  'python_version': platform.python_version(), 'platform': platform.platform(),
                  'model_sha256': sha256(path)}
        write_json(path.with_suffix('.metadata.json'), record)
        # Verify actual artifact inference before test access; reload time is not scored.
        reloaded = predict_frame(path, validation.iloc[:50], trusted=True)
        if not np.allclose(reloaded, predictions[:50], rtol=1e-6, atol=1e-6):
            raise RuntimeError('Saved-model inference differs from in-memory model')
        records[name] = record
        pd.DataFrame(rows).to_csv(output / 'validation_results.csv', index=False)
        print(f'{name}: validation MAE={row["MAE"]:.6f}, RMSE={row["RMSE"]:.6f}, training={training_seconds:.2f}s', flush=True)
    selected = select_model(rows)
    lock = {'selected_model': selected, 'selection_metric': 'validation MAE',
            'locked_utc': datetime.now(timezone.utc).isoformat(), 'validation_results': rows.copy(),
            'frozen_model_hashes': {name: record['model_sha256'] for name, record in records.items()},
            'test_used_for_selection': False}
    write_json(output / 'selection_lock.json', lock)
    print(f'Selection locked: {selected}. Opening test partition for final evaluation only.', flush=True)
    test = read_partitions(dataset, ['test'], manifest)
    X_test, y_test = checked_features(test, features), test.target_value.to_numpy()
    ranges['test'] = range_metadata(test)
    station_rows = []
    predictions_table = test[['forecast_origin', 'target_start', 'target_end', 'district', 'port', 'target_value']].copy()
    for name, model in fitted.items():
        row, prediction = score(model, X_test, y_test, name, 'test', records[name]['training_seconds'])
        rows.append(row)
        predictions_table[name] = prediction
        for (district, port), group in test.groupby(['district', 'port'], observed=True):
            positions = test.index.get_indexer(group.index)
            station_rows.append({'model': name, 'district': district, 'port': port, 'rows': len(group),
                                 **regression_metrics(group.target_value, prediction[positions])})
        records[name]['dataset_range'] = ranges
        records[name]['evaluation_results']['test'] = row
        records[name]['selected_by_validation'] = name == selected
        records[name]['final_evaluation_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(models_dir / f'{name}.metadata.json', records[name])
    comparison = pd.DataFrame(rows)
    comparison['selected_by_validation'] = comparison.model.eq(selected)
    comparison.to_csv(output / 'model_comparison.csv', index=False)
    pd.DataFrame(station_rows).to_csv(output / 'test_station_metrics.csv', index=False)
    predictions_table.to_csv(output / 'final_test_predictions.csv.gz', index=False, float_format='%.9g')
    after = fingerprint(root)
    if before != after or sha256(dataset) != manifest['dataset_sha256']:
        raise RuntimeError('Input fingerprint changed during training')
    write_json(output / 'run_manifest.json', {'started_utc': contract['created_utc'],
               'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_before': before,
               'source_after': after, 'sources_unchanged': True, 'feature_dataset_unchanged': True,
               'selected_model': selected, 'versions': versions, 'dataset_ranges': ranges,
               'test_evaluation_passes': 1, 'rows': len(development) + len(test),
               'model_hashes': lock['frozen_model_hashes']})
    write_json(models_dir / 'selected_model.json', {'model': selected, 'path': f'{selected}.joblib',
               'selection': 'validation MAE; frozen before test evaluation', 'model_sha256': records[selected]['model_sha256']})
    (output / 'model_evaluation.md').write_text(make_report(comparison, selected, ranges, versions, contract), encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    return comparison


def make_report(comparison, selected, ranges, versions, contract):
    summary = comparison[['model', 'split', 'rows', 'MAE', 'RMSE', 'R2', 'training_seconds', 'prediction_seconds']].to_string(index=False, float_format=lambda x: f'{x:.6f}')
    selected_test = comparison.loc[comparison.model.eq(selected) & comparison.split.eq('test')].iloc[0]
    return f'''# Phase 7: machine learning evaluation

## Scientific target and scope

TARGET = continuous hourly PM2.5 mean (`target_value`), in the original source numeric scale.
PM2.5 is populated and supported by complete hourly observations. Physical units remain unverified,
so this is a provisional concentration prediction experiment, not verified AQI or a health advisory.
AQI is not selected: required unit verification and regulatory averaging eligibility are unresolved.
FEATURES = the 187 numeric Phase 6 predictors, listed below and in each model metadata file.
FORECAST HORIZON = at local origin t predict the mean over [t,t+1 hour).
Measurement features end strictly BEFORE t; lag 1 uses [t-2h,t-1h), not the hour ending at t.
There is thus a conservative one-hour observation-availability gap. Unknown real sensor publication
latency must be checked before deployment. Source naive times are assumed Asia/Kolkata.

## Preprocessing and temporal evaluation

Phase 6 train-only medians, missingness flags and location vocabularies are reused unchanged.
No random split, shuffled temporal validation, future weather, target imputation or test-set tuning.
All models receive the same eligible rows. PM2.5 history requires >=18 valid hours in the prior 24h.
Feature dataset SHA-256: `{contract['dataset_sha256']}`.
The feature schema alone is X; labels, split, timestamps and raw district/port strings are excluded.
Pooled results weight each observation equally; station-level test metrics are also saved, since
station coverage and pollutant distributions differ. These are existing-station temporal tests,
not evidence of generalization to entirely new monitoring locations.

Dataset ranges:
```
{json.dumps(ranges, indent=2)}
```
Targets ending on/crossing the next split boundary were purged by Phase 6. Within validation/test,
earlier actual observations may become predictors of later origins. This evaluates rolling-origin
operational forecasting, NOT a fixed-origin multi-step prediction. No refit on train+validation was
performed: saved models are the exact train-only candidates selected on validation.

## Prespecified candidates and selection

1. Persistence first: observed PM2.5 lag 1, with TRAIN target median fallback when missing.
2. Random Forest: 80 trees, depth 14, minimum leaf 5, feature fraction .7, bootstrap fraction .8.
3. XGBoost: 250 trees, depth 6, learning rate .05, child weight 5, row/column fractions .8,
   L2=1, squared-error objective, CPU histogram method. Both learned models use seed 42.

Parameters were fixed before scoring; no hyperparameter search or early stopping occurred.
Bootstrap/subsampling is restricted to training data and does not alter chronological partitions.
Negative predictions are clipped to zero for every model, a rule fixed before validation.
Selection minimizes validation MAE (ties: validation RMSE then model name), including baseline.
**Selected model: {selected}.** Selection was written to `selection_lock.json` before loading
test rows for scoring. All frozen candidates received one final test pass for transparent comparison;
no model or parameter was changed or reselected based on those results. Repeated runner execution
is blocked once models/selection have been saved. Future development needs a new untouched holdout.

## Results

```
{summary}
```
Selected model final test: MAE={selected_test.MAE:.6f}, RMSE={selected_test.RMSE:.6f}, R²={selected_test.R2:.6f}.
MAE = mean absolute error; RMSE = square root of mean squared error;
R² = 1 - sum squared errors / sum squared deviations of actual targets from their mean.
R² can be negative and is undefined for constant targets (exported blank/null).
MAE/RMSE use the original numeric scale, not verified micrograms/m³. R² is dimensionless.
Training seconds time fit only; prediction seconds time one batch predict plus zero-floor processing,
excluding data loading, serialization and metric calculation. These are single-run wall-clock timings,
not robust latency benchmarks; per-row batch timing is not real-time endpoint latency.

## Saved artifacts and inference

`ML/saved_models/` contains each trained .joblib model and accompanying .metadata.json,
plus selected_model.json. Metadata records UTC training date, full feature order, target/horizon,
dataset ranges/hash, algorithm/parameters, training and prediction timings, validation/test results,
runtime versions, frozen preprocessing state and model checksum.
`model_comparison.csv` holds all requested metrics and times; `test_station_metrics.csv` exposes
location-level performance; `final_test_predictions.csv.gz` supports error auditing.
`experiment_protocol.json` and `selection_lock.json` document the evaluation sequence.

Run training from project root: `python -m ML.model_evaluation --n-jobs 2`.
Inference: `python -m ML.predict --model ML/saved_models/{selected}.joblib --input features.csv --output predictions.csv --trusted-model`.
Input must be Phase 6 engineered rows built with saved preprocessing state, not raw measurements.
Feature order is enforced; missing/nonfinite engineered columns are rejected. Extra metadata is ignored.
The upstream feature builder must enforce historical cutoff and coverage; a numeric feature table alone
cannot prove real-world publication availability. No interval uncertainty or operational service is provided.
Joblib files can execute code: use only trusted model artifacts; checksums do not establish trust.

Runtime versions: {json.dumps(versions)}. See `ML/requirements.txt` for reproducible dependencies.
For this workspace, ML-only dependencies were installed under `.runtime/ml`; in PowerShell set
`$env:PYTHONPATH = (Resolve-Path .runtime/ml).Path` before invoking the bundled Python executable.
The model modules never import DATA_STRUCTURES; custom structures remain independent.
Raw/processed source fingerprints and the Phase 6 dataset hash were verified unchanged.

## Limitations

One chronological holdout is not a stability/confidence-interval study. Missingness and the history
filter mean results do not cover all station-hours. Unit metadata, source timestamp interval semantics
and publication delays require external verification before scientific or operational deployment.
Source labels and measured concentration distributions can differ across stations. No causal weather
claims, health conclusions, spatial extrapolation, LSTM or anomaly/clustering models are implemented.
Implementation stopped after Phase 7; no website/backend work performed.

## Complete predictor list

```
{chr(10).join(contract['FEATURES'])}
```

## Implementation references

- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--n-jobs', type=int, default=2)
    args = parser.parse_args()
    run(args.root, args.n_jobs)
