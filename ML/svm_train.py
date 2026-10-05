"""Controlled future-category experiment reusing Phase 6 features and time splits.

Predeclared bounded training budget shared by all candidate kernels. No synthetic
oversampling, shuffled temporal splits, refit on test, or repeated test selection.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from FEATURE_ENGINEERING.build_features import temporal_split
from ML.model_evaluation import sha256, write_json, range_metadata
from ML.svm_classifier import create_model, classification_metrics, target_classes
from ML.svm_predict import validate_features
from EDA.common import fingerprint

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [('linear_C0.1', 'linear', .1, 'scale'), ('linear_C1', 'linear', 1., 'scale'),
              ('rbf_C1', 'rbf', 1., 'scale'), ('rbf_C10', 'rbf', 10., 'scale')]
TRAINING_CAP = 6000


def choose_training_rows(frame, cap=TRAINING_CAP):
    """Deterministic, approximately proportional per-class time-spread sample.

Rare classes keep at least up to 10 actual observations; never duplicate rows.
Returned observations remain chronological. The cap is a compute budget, not a
claim that the full eligible cohort was fitted. Validation/test are never sampled.
    """
    if len(frame) <= cap:
        return frame.copy()
    indices = []
    for _, group in frame.groupby('category', sort=True):
        count = min(len(group), max(10, int(cap * len(group) / len(frame))))
        indices.extend(group.iloc[np.linspace(0, len(group)-1, count, dtype=int)].index)
    return frame.loc[sorted(set(indices))].sort_values(['forecast_origin', 'district', 'port'])


def prepare(root):
    feature_dir = root / 'REPORTS/FEATURE_ENGINEERING'
    state = json.loads((feature_dir / 'preprocessing_state.json').read_text())
    manifest = json.loads((feature_dir / 'run_manifest.json').read_text())
    label_summary = json.loads((root / 'REPORTS/AQI/label_summary.json').read_text())
    path = feature_dir / 'ml_ready_dataset.csv.gz'
    labels_path = root / 'REPORTS/AQI/verified_aqi_labels.csv.gz'
    if sha256(path) != manifest['dataset_sha256'] or sha256(labels_path) != label_summary['labels_sha256']:
        raise ValueError('Feature/label checksum mismatch')
    if fingerprint(root) != manifest['source_after']:
        raise ValueError('Current source data differ from the audited feature archive')
    # Only physically identified pollutants; retain existing weather/calendar/location
    # predictors and train-only imputation/missingness contracts unchanged.
    features = [n for n in state['feature_columns'] if not n.startswith(('nox_', 'benzene_', 'toluene_', 'xylene_'))]
    forbidden = {'aqi', 'category', 'target_value', 'target_start', 'target_end', 'forecast_origin', 'split', 'district', 'port'}
    if forbidden.intersection(features):
        raise ValueError('Future target or metadata in feature list')
    metadata_cols = ['forecast_origin', 'target_start', 'target_end', 'district', 'port', 'split']
    frame = pd.read_csv(path, usecols=metadata_cols+features,
                        dtype={f: 'float32' for f in features}, parse_dates=metadata_cols[:3])
    if not frame.forecast_origin.is_monotonic_increasing:
        raise ValueError('Feature rows not chronological')
    if not frame.split.eq(temporal_split(frame)).all():
        raise ValueError('Feature temporal split mismatch')
    if not (frame.target_end-frame.forecast_origin).eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Unsupported forecast horizon')
    labels = pd.read_csv(labels_path, usecols=['district', 'port', 'label_end', 'category'], parse_dates=['label_end'])
    if labels.duplicated(['district', 'port', 'label_end']).any():
        raise ValueError('Duplicate target station/time')
    joined = frame.merge(labels, left_on=['district', 'port', 'target_end'],
                         right_on=['district', 'port', 'label_end'], how='left', validate='one_to_one')
    missing = int(joined.category.isna().sum())
    joined = joined.loc[joined.category.notna()].sort_values(['forecast_origin', 'district', 'port']).reset_index(drop=True)
    # A true past-category baseline: latest permitted category ends at t-1h,
    # never at the future target or the excluded hour ending at origin t.
    history = labels.rename(columns={'label_end': 'baseline_end', 'category': 'past_category'})
    joined['baseline_end'] = joined.forecast_origin - pd.Timedelta(hours=1)
    joined = joined.merge(history, on=['district', 'port', 'baseline_end'], how='left', validate='one_to_one')
    return joined, features, state, manifest, label_summary, missing


def run(root=ROOT):
    import joblib
    root = Path(root)
    output = root / 'REPORTS/ML/svm'
    models = root / 'ML/saved_models'
    if (models / 'svm_model.joblib').exists() or (output / 'selection_lock.json').exists():
        raise ValueError('SVM experiment already locked; do not repeatedly tune against test results')
    output.mkdir(parents=True, exist_ok=True)
    before = fingerprint(root)
    protected = {p.relative_to(root).as_posix(): sha256(p) for p in models.rglob('*') if p.is_file()}
    frame, features, state, manifest, labels_summary, missing = prepare(root)
    train = frame.loc[frame.split.eq('train')].copy()
    validation = frame.loc[frame.split.eq('validation')].copy()
    test = frame.loc[frame.split.eq('test')].copy()
    fitted = choose_training_rows(train)
    if train.category.nunique() < 2 or min(len(train), len(validation), len(test)) == 0:
        raise ValueError('Insufficient chronological samples/classes')
    distribution = []
    for name, partition in [('train_eligible', train), ('train_fitted', fitted), ('validation', validation), ('test', test)]:
        counts = partition.category.value_counts()
        for category in target_classes():
            distribution.append({'split': name, 'category': category, 'count': int(counts.get(category, 0)),
                                 'fraction': float(counts.get(category, 0)/len(partition))})
    pd.DataFrame(distribution).to_csv(output / 'class_distribution.csv', index=False)
    protocol = {'target': 'AQI category at window_end = forecast_origin + 1 hour', 'features': features,
        'horizon_hours': 1, 'measurement_cutoff': 'all observations end strictly before origin t; latest end t-1h',
        'target_window': '[t-23h,t+1h), with engine-specific pollutant averaging and coverage',
        'split': {'train': '2023–2024', 'validation': '2025', 'test': '2026; retrospective reused period'},
        'selection': 'Highest validation macro F1 over all six classes; then accuracy; then candidate id',
        'training_cap': TRAINING_CAP, 'sampling': 'Deterministic proportional within-class time-spread, preserve up to 10 rare examples; no duplication; sorted chronologically',
        'candidates': CANDIDATES, 'probabilities': False, 'test_access': 'Class support audited before training; no test prediction/metric until selection lock',
        'feature_preprocessing': 'Reuse Phase 6 train-only medians/one-hot vocabulary, then fit StandardScaler on fitted training rows only',
        'source_fingerprint': before, 'labels_sha256': labels_summary['labels_sha256']}
    write_json(output / 'protocol.json', protocol)
    print('Pre-training class distribution:\n' + pd.DataFrame(distribution).to_string(index=False), flush=True)
    X, y = validate_features(fitted[features], features), fitted.category
    Xv, yv = validate_features(validation[features], features), validation.category
    baseline = DummyClassifier(strategy='most_frequent').fit(train[features], train.category)
    majority = train.category.mode().iloc[0]
    write_json(output / 'baseline_validation.json', {
        'majority': classification_metrics(yv, baseline.predict(Xv)),
        'past_category': classification_metrics(yv, validation.past_category.fillna(majority))})
    comparisons, candidates = [], {}
    for name, kernel, C, gamma in CANDIDATES:
        print(f'Training {name}: {len(X)} rows, {len(features)} features', flush=True)
        model = create_model(kernel, C, gamma)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X, y)
        training_seconds = perf_counter()-start
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise ValueError(f'{name} did not converge; stop rather than select an unfinished fit')
        start = perf_counter()
        prediction = model.predict(Xv)
        prediction_seconds = perf_counter()-start
        metrics = classification_metrics(yv, prediction)
        comparisons.append({'candidate': name, 'kernel': kernel, 'C': C, 'gamma': gamma if kernel == 'rbf' else None,
                            'training_seconds': training_seconds, 'prediction_seconds': prediction_seconds,
                            **{k: v for k, v in metrics.items() if isinstance(v, float)}})
        candidates[name] = (model, metrics)
        print(f'{name}: validation macro F1={metrics["macro_f1"]:.4f}', flush=True)
    selected = sorted(comparisons, key=lambda row: (-row['macro_f1'], -row['accuracy'], row['candidate']))[0]
    model, validation_metrics = candidates[selected['candidate']]
    write_json(output / 'selection_lock.json', {'selected': selected, 'selected_at': datetime.now(timezone.utc).isoformat(),
                                               'validation_only': True, 'test_evaluated': False})
    pd.DataFrame(comparisons).to_csv(output / 'validation_candidates.csv', index=False)
    # Final test evaluation begins only after the lock. No model is refitted.
    Xt, yt = validate_features(test[features], features), test.category
    start = perf_counter()
    predicted = model.predict(Xt)
    seconds = perf_counter()-start
    metrics = classification_metrics(yt, predicted)
    baseline_metrics = classification_metrics(yt, baseline.predict(Xt))
    persistence_metrics = classification_metrics(yt, test.past_category.fillna(majority))
    model_path = models / 'svm_model.joblib'
    joblib.dump(model, model_path)
    metadata = {'model_name': 'Support Vector Machine', 'model_type': 'Classification',
        'purpose': 'Future AQI Category Classification', 'target': 'aqi_category', 'target_classes': target_classes(),
        'learned_classes': model.classes_.tolist(), 'features': features,
        'training_date_utc': datetime.now(timezone.utc).isoformat(), 'forecast_horizon_hours': 1,
        'sampling_interval_minutes': 60, 'dataset_range': {n: range_metadata(v) for n, v in [('train', train), ('validation', validation), ('test', test), ('fitted_training', fitted)]},
        'kernel': selected['kernel'], 'C': selected['C'], 'gamma': selected['gamma'],
        'parameters': model.named_steps['svm'].get_params(), 'class_distribution': distribution,
        'preprocessing': {'pipeline': 'StandardScaler -> SVM; serialized together', 'fitted_scaler_rows': int(model.named_steps['scaler'].n_samples_seen_),
                          'feature_preprocessing_state': state, 'feature_state_sha256': sha256(root / 'REPORTS/FEATURE_ENGINEERING/preprocessing_state.json')},
        'evaluation_results': {'validation': validation_metrics, 'test': metrics, 'majority_test': baseline_metrics, 'past_category_test': persistence_metrics},
        'training_seconds': selected['training_seconds'], 'prediction_seconds': seconds,
        'probabilities_enabled': False, 'calibrated_confidence_available': False,
        'model_sha256': sha256(model_path), 'source_fingerprint': before,
        'label_summary': labels_summary, 'unit_registry_sha256': sha256(root / 'CONFIG/pollutant_units.json'),
        'feature_dataset_sha256': manifest['dataset_sha256'], 'feature_rows_without_valid_category': missing,
        'stations': frame[['district', 'port']].drop_duplicates().to_dict('records'), 'protocol': protocol,
        'limitations': 'Source website identity/scope is user-attested. Overlapping AQI windows are autocorrelated. Historical 2026 period was used in prior regression phases; retrospective, not a new blind holdout. Predictions are not observed AQI.'}
    write_json(models / 'svm_metadata.json', metadata)
    reports = []
    for split, scores in [('validation', validation_metrics), ('test', metrics)]:
        reports += [{'split': split, **row} for row in scores['per_class']]
    pd.DataFrame(reports).to_csv(root / 'REPORTS/ML/svm_classification_report.csv', index=False)
    pd.DataFrame(metrics['confusion_matrix'], index=target_classes(), columns=target_classes()).to_csv(output / 'confusion_matrix.csv', index_label='Actual')
    prediction_rows = test[['forecast_origin', 'target_end', 'district', 'port', 'category']].copy()
    prediction_rows['predicted_category'] = predicted
    prediction_rows.to_csv(output / 'test_predictions.csv.gz', index=False)
    example = test.iloc[0]
    write_json(output / 'example_request.json', {'features': {f: float(example[f]) for f in features},
        'district': example.district, 'port': example.port, 'forecast_origin': example.forecast_origin.isoformat(),
        'data_source': 'HISTORICAL_EVALUATION'})
    write_json(output / 'example_expected.json', {'actual_future_category': example.category,
        'predicted_category': str(predicted[0]), 'target_end': example.target_end.isoformat(), 'hard_coded_application_label': False})
    from SCRIPTS.finalize_phase10a import plot_confusion
    plot_confusion(root, metrics['confusion_matrix'])
    after = fingerprint(root)
    if before != after or any(sha256(root / p) != digest for p, digest in protected.items()):
        raise RuntimeError('Earlier source/model artifacts changed')
    write_json(output / 'run_manifest.json', {'source_before': before, 'source_after': after,
        'earlier_model_hashes': protected, 'earlier_models_unchanged': True, 'selected': selected,
        'test_metrics': {k: v for k, v in metrics.items() if isinstance(v, float)}})
    print(json.dumps({'selected': selected, 'test': metrics, 'majority_macro_f1': baseline_metrics['macro_f1'],
                      'past_category_macro_f1': persistence_metrics['macro_f1']}, indent=2), flush=True)


if __name__ == '__main__':
    run()
