"""Phase 9 chronological matched-input benchmark and LSTM training.

Run python -B -m ML.lstm_train. Existing models/reports remain frozen. This is
a retrospective comparison on the previously examined 2026 period, not a new
blind holdout. Model/epoch selection uses 2025 validation only.
"""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from EDA.common import fingerprint
from ML.sequence_builder import FEATURES, build_sequences, load_hourly
from ML.model_evaluation import regression_metrics, sha256, write_json


def predict_batches(model, values, scaler, batch_size=1024):
    import torch
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            results.append(model(torch.from_numpy(values[start:start+batch_size])).numpy())
    return np.maximum(0, np.concatenate(results)*scaler['target_std'] + scaler['target_mean'])


def run(root, length=24, epochs=12):
    import torch
    import joblib
    import sklearn
    from ML.lstm_model import PollutantLSTM
    from ML.random_forest import create_model as forest
    from ML.xgboost_model import create_model as xgb
    root = Path(root).resolve()
    output, saved = root / 'REPORTS/ML/phase9', root / 'ML/saved_models'
    if (output / 'selection_lock.json').exists() or (saved / 'lstm_model.pt').exists():
        raise ValueError('Phase 9 already trained/locked; do not repeatedly tune on the retrospective test')
    output.mkdir(parents=True, exist_ok=True)
    benchmarks = saved / 'phase9_comparators'
    benchmarks.mkdir(parents=True, exist_ok=True)
    protected = [p for p in (root / 'REPORTS/ML').rglob('*') if p.is_file() and 'phase9' not in p.parts and not p.name.startswith('lstm_')]
    protected += [p for p in saved.rglob('*') if p.is_file() and 'phase9_comparators' not in p.parts and not p.name.startswith('lstm_')]
    previous = {str(p.relative_to(root)): sha256(p) for p in protected}
    before = fingerprint(root)
    hourly = load_hourly(root)
    X, y, meta, audit = build_sequences(hourly, length)
    audit.to_csv(output / 'sequence_continuity_audit.csv', index=False)
    train = meta.split.eq('train').to_numpy()
    val = meta.split.eq('validation').to_numpy()
    test = meta.split.eq('test').to_numpy()
    if min(train.sum(), val.sum(), test.sum()) < 1000:
        raise ValueError('Insufficient temporal sequence support for this configured experiment')
    print(f'Eligible sequences: train={train.sum()}, validation={val.sum()}, test={test.sum()}', flush=True)
    # Statistics are computed only from TRAIN windows/targets, never whole data.
    mean, std = X[train].mean(axis=(0, 1), dtype=np.float64), X[train].std(axis=(0, 1), dtype=np.float64)
    std[std < 1e-8] = 1
    scaler = {'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
              'target_mean': float(y[train].mean()), 'target_std': float(y[train].std())}
    write_json(saved / 'lstm_scaler.json', scaler)
    scaled = ((X-mean)/std).astype('float32')
    target_scaled = ((y[train]-scaler['target_mean'])/scaler['target_std']).astype('float32')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    model = PollutantLSTM(len(FEATURES), 32, std[0]/scaler['target_std'], (mean[0]-scaler['target_mean'])/scaler['target_std'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = torch.nn.MSELoss()
    Xtrain = scaled[train]
    best_mae, best_weights, best_epoch, stale = float('inf'), None, 0, 0
    history = []
    started = datetime.now(timezone.utc).isoformat()
    protocol = {'target': 'pm25', 'horizon_hours': 1, 'sequence_length': length, 'features': FEATURES,
                'maximum_epochs': epochs, 'early_stop_patience': 3, 'batch_size': 512, 'shuffle': False,
                'selection': 'validation MAE only', 'test_status': '2026 previously examined in Phase 7; retrospective benchmark only',
                'common_information': 'All four primary methods use same complete 24-hour pollutant/calendar sequence and rows',
                'created_utc': started}
    write_json(output / 'protocol.json', protocol)
    start = perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        for index in range(0, len(Xtrain), 512):
            batch = torch.from_numpy(Xtrain[index:index+512])
            target = torch.from_numpy(target_scaled[index:index+512])
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(batch)
        prediction = predict_batches(model, scaled[val], scaler)
        metrics = regression_metrics(y[val], prediction)
        history.append({'epoch': epoch, 'train_scaled_mse': total/len(Xtrain), **metrics})
        print(f'LSTM epoch {epoch}: validation MAE={metrics["MAE"]:.5f}', flush=True)
        if metrics['MAE'] < best_mae - 1e-5:
            best_mae, best_weights, best_epoch, stale = metrics['MAE'], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= 3:
            break
    lstm_seconds = perf_counter()-start
    model.load_state_dict(best_weights)
    model.eval()
    pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
    # Fair primary comparator retrains: identical sequence rows and information.
    flat = X.reshape(len(X), -1)
    estimators, times, rows = {}, {'persistence': 0., 'lstm': lstm_seconds}, []
    for name, estimator in [('random_forest', forest(2)), ('xgboost', xgb(2))]:
        print(f'Training matched-input {name}', flush=True)
        start = perf_counter()
        estimator.fit(flat[train], y[train])
        times[name] = perf_counter()-start
        estimators[name] = estimator
        joblib.dump(estimator, benchmarks / f'{name}.joblib', compress=3)
    for name in ('persistence', 'random_forest', 'xgboost', 'lstm'):
        start = perf_counter()
        prediction = X[val, -1, 0] if name == 'persistence' else predict_batches(model, scaled[val], scaler) if name == 'lstm' else np.maximum(0, estimators[name].predict(flat[val]))
        elapsed = perf_counter()-start
        rows.append({'model': name, 'split': 'validation', 'rows': int(val.sum()), **regression_metrics(y[val], prediction),
                     'training_seconds': times[name], 'prediction_seconds': elapsed})
    chosen = min(rows, key=lambda row: (row['MAE'], row['RMSE'], row['model']))['model']
    write_json(output / 'selection_lock.json', {'selected_by_validation': chosen, 'best_lstm_epoch': best_epoch,
               'validation_results': rows.copy(), 'test_used_for_selection': False, 'locked_utc': datetime.now(timezone.utc).isoformat()})
    torch.save(model.state_dict(), saved / 'lstm_model.pt')
    print(f'Selection frozen: {chosen}; retrospective 2026 comparison begins', flush=True)
    predictions = meta.loc[test].copy()
    predictions['actual_pm25'] = y[test]
    for name in ('persistence', 'random_forest', 'xgboost', 'lstm'):
        start = perf_counter()
        prediction = X[test, -1, 0] if name == 'persistence' else predict_batches(model, scaled[test], scaler) if name == 'lstm' else np.maximum(0, estimators[name].predict(flat[test]))
        elapsed = perf_counter()-start
        predictions[name] = prediction
        rows.append({'model': name, 'split': 'test_retrospective', 'rows': int(test.sum()), **regression_metrics(y[test], prediction),
                     'training_seconds': times[name], 'prediction_seconds': elapsed})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'model_comparison.csv', index=False)
    predictions.to_csv(root / 'REPORTS/ML/lstm_predictions.csv', index=False, float_format='%.9g')
    # Original Phase 7 benchmark: reuse saved predictions, never re-evaluate models.
    existing = pd.read_csv(root / 'REPORTS/ML/final_test_predictions.csv.gz', parse_dates=['forecast_origin'])
    matched = predictions.merge(existing, on=['forecast_origin', 'district', 'port'], suffixes=('', '_phase7'), validate='one_to_one')
    original_rows = []
    for name in ('persistence', 'random_forest', 'xgboost'):
        original_rows.append({'model': name + '_phase7_frozen', 'rows': len(matched), **regression_metrics(matched.actual_pm25, matched[name + '_phase7'])})
    original_rows.append({'model': 'lstm_same_intersection', 'rows': len(matched), **regression_metrics(matched.actual_pm25, matched.lstm)})
    pd.DataFrame(original_rows).to_csv(output / 'phase7_frozen_comparison.csv', index=False)
    counts = meta.groupby('split').agg(rows=('forecast_origin', 'size'), first_origin=('forecast_origin', 'min'), last_origin=('forecast_origin', 'max')).reset_index()
    counts.to_csv(output / 'split_summary.csv', index=False)
    station_list = [{'district': str(d), 'port': str(p)} for d, p in hourly[['district', 'port']].drop_duplicates().itertuples(index=False, name=None)]
    metadata = {'model_id': 'pm25-lstm-' + sha256(saved / 'lstm_model.pt')[:12], 'training_date_utc': started,
                'target': 'pm25', 'units': 'original numeric scale; physical units unverified', 'features': FEATURES,
                'observation_features': ['pm25'], 'sequence_length': length, 'sampling_interval_minutes': 60,
                'forecast_horizon_hours': 1, 'supported_horizons_hours': [1], 'availability_gap_hours': 1,
                'timestamp_semantics': 'local hourly interval starts; Asia/Kolkata assumed',
                'architecture': {'type': 'residual LSTM', 'hidden_units': 32, 'layers': 1, 'output_units': 1},
                'training': {**protocol, 'best_epoch': best_epoch, 'optimizer': 'Adam', 'learning_rate': .001, 'seed': 42},
                'input_validation': {'pm25': {'minimum': 0., 'maximum': float(X[train, :, 0].max()), 'meaning': 'training support bound, not health threshold'}},
                'stations': station_list, 'selected_primary_model': chosen, 'evaluation': rows,
                'model_sha256': sha256(saved / 'lstm_model.pt'), 'scaler_sha256': sha256(saved / 'lstm_scaler.json'),
                'source_fingerprint': before, 'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
                'real_time_feed': False, 'aqi_applicable': False,
                'aqi_unavailable_reason': 'Single-hour PM2.5 prediction, unverified units and insufficient eligible pollutants/averaging; no official AQI/category/health alert.'}
    write_json(saved / 'lstm_metadata.json', metadata)
    # Saved-weight round trip without an additional test pass.
    restored = PollutantLSTM()
    restored.load_state_dict(torch.load(saved / 'lstm_model.pt', map_location='cpu', weights_only=True))
    np.testing.assert_allclose(predict_batches(restored, scaled[val][:32], scaler), predict_batches(model, scaled[val][:32], scaler))
    after = fingerprint(root)
    if before != after or previous != {str(p.relative_to(root)): sha256(p) for p in protected}:
        raise RuntimeError('Source or completed earlier-phase artifacts changed')
    write_json(output / 'run_manifest.json', {'sources_unchanged': True, 'prior_ml_artifacts_unchanged': True,
               'source_before': before, 'source_after': after, 'protected_hashes': previous,
               'selected_model': chosen, 'best_lstm_epoch': best_epoch, 'completed_utc': datetime.now(timezone.utc).isoformat()})
    report = f'''# Phase 9 LSTM evaluation

TARGET = next hourly PM2.5 mean in original source scale; units unverified, NOT AQI.
FEATURES = PM2.5 plus six derived cyclical calendar features per time step: {', '.join(FEATURES)}.
SEQUENCE = {length} consecutive valid hourly observations. HORIZON = 1 hour at origin t,
predicting [t,t+1h); newest input is [t-2h,t-1h). One-hour availability gap matches Phase 7.

## Continuity and eligibility

Native inputs are audited 15/60-minute files aggregated to hourly only with complete native
coverage. sequence_continuity_audit.csv records actual gaps, usable runs and windows by station.
No gaps are compressed into adjacent rows, no imputation fills sequences, and windows never cross
stations. The longest valid run is {int(audit.longest_valid_run_hours.max())} hours. Eligibility is
substantial but uneven: two stations have no eligible training sequences; the pooled model does
not use station IDs. Treat their later predictions as cross-station application, not in-station validation.

```
{counts.to_string(index=False)}
```

## Training and comparison

One unidirectional 32-unit LSTM, linear residual over persistence, Adam .001, MSE loss,
gradient norm cap 1, batch 512, no shuffling, seed 42, max {epochs} epochs, patience 3.
Epoch {best_epoch} selected using validation MAE. Feature and target scalers fit ONLY training data;
weights and JSON scalers saved for inference. Overlapping windows are dependent observations, not
independent samples. No claim of statistical significance or confidence intervals is made.

Primary comparison retrains the existing RF/XGBoost algorithm configurations on identical complete
sequence rows and flattened sequence inputs (same information as LSTM). Persistence uses the newest
eligible PM2.5 hour. All methods use identical target, dates, horizon and zero-floor output policy.
No method gets unavailable meteorology or future values. Validation-selected method: **{chosen}**.
No winner is chosen from retrospective test scores. No extra horizon is claimed or recursively
generated. Random row subsampling internal to tree fitting is restricted to training data.

```
{comparison.to_string(index=False)}
```

MAE/RMSE are source-scale errors; R² is dimensionless. LSTM training time includes validation
checks/early stopping; tree timings cover fit only. Prediction times cover batched computation,
not server/network latency. Hardware-dependent single-run timings are not latency guarantees.

## Previously examined test period

The 2026 dates were already evaluated in Phase 7. This is therefore a transparent retrospective
comparison, NOT a fresh blind holdout or proof of prospective performance. No 2026 score influenced
this run's epoch, parameter or winner selection. Prospective evaluation requires newly collected data.
Original Phase 7 models and predictions were left unchanged. A secondary comparison joins their
saved predictions to exactly matching station/origin rows (different feature budgets/training cohorts,
so not a controlled architecture-only comparison):

```
{pd.DataFrame(original_rows).to_string(index=False)}
```

## Saved outputs

REPORTS/ML/lstm_predictions.csv contains timestamp/origin, district, station, actual PM2.5 and
each method's prediction. phase9/model_comparison.csv, training_history.csv, split_summary.csv,
sequence_continuity_audit.csv, phase7_frozen_comparison.csv and selection_lock.json provide audit detail.
ML/saved_models/lstm_model.pt, lstm_metadata.json and lstm_scaler.json are the reusable LSTM artifacts.
The primary winner is not automatically the LSTM; the LSTM remains available as the requested
experimental module. Phase 7's selected-model pointer is not replaced.

Forecasts are historical or manually/synthetically supplied. No live sensor/API feed exists.
AQI/category/health alerts remain unavailable for this single-pollutant/hourly unverified-unit output.
'''
    (root / 'REPORTS/ML/lstm_evaluation.md').write_text(report, encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    print(f'Validation-selected: {chosen}', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-length', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 50:
        raise ValueError('epochs must be 1..50')
    run(Path(__file__).resolve().parents[1], args.sequence_length, args.epochs)
