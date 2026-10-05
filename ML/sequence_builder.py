"""Gap-aware PM2.5 sequences. No imputation, cross-station windows or shuffling."""
from pathlib import Path
import numpy as np
import pandas as pd
from FEATURE_ENGINEERING.build_features import temporal_split
from ML.unsupervised_data import validate_hourly

FEATURES = ['pm25', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']


def encode_observations(timestamps, pm25):
    """Return pollutant + known calendar inputs; timestamps are local interval starts."""
    t = pd.DatetimeIndex(timestamps)
    return np.column_stack([pm25, np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
                            np.sin(2*np.pi*t.dayofweek/7), np.cos(2*np.pi*t.dayofweek/7),
                            np.sin(2*np.pi*(t.month-1)/12), np.cos(2*np.pi*(t.month-1)/12)]).astype('float32')


def build_sequences(hourly, length=24, horizon=1):
    """Predict [origin,origin+1h) from hourly starts origin-L-1 ... origin-2.

    The newest observation ENDS origin-1h, strictly before origin, matching Phase
    7. Dense reindexing makes 24 values actually span 24 consecutive hours. Return
    X [examples,length,7], y, chronological metadata, per-station continuity audit.
    Only supported one-hour horizon accepted; length configurable for experiments.
    """
    if type(length) is not int or not 2 <= length <= 168 or horizon != 1:
        raise ValueError('Sequence length must be 2..168; only the evaluated 1h horizon is supported')
    frame = validate_hourly(hourly, ['pm25'])
    arrays, targets, metadata, audits = [], [], [], []
    for (district, port), group in frame.groupby(['district', 'port'], observed=True):
        index = pd.date_range(group.timestamp.min(), group.timestamp.max(), freq='h')
        series = group.set_index('timestamp').pm25.reindex(index)
        valid = series.notna().to_numpy()
        padded = np.r_[False, valid, False].astype(int)
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        runs = ends - starts
        audit = {'district': str(district), 'port': str(port), 'grid_hours': len(index),
                 'observed_hour_bins': len(group), 'missing_entire_hours': len(index)-len(group),
                 'usable_pm25_hours': int(valid.sum()), 'missing_pm25_grid_hours': int((~valid).sum()),
                 'continuous_valid_runs': len(runs), 'longest_valid_run_hours': int(runs.max()) if len(runs) else 0,
                 'runs_at_least_sequence_length': int((runs >= length).sum())}
        if len(index) <= length+1:
            audits.append(audit)
            continue
        values = series.to_numpy(dtype=float)
        encoded = encode_observations(index, values)
        windows = np.lib.stride_tricks.sliding_window_view(encoded, length, axis=0).swapaxes(1, 2)[:-2]
        y = values[length+1:]
        good = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(y)
        origins = index[length+1:][good]
        meta = pd.DataFrame({'forecast_origin': origins, 'target_start': origins,
                             'target_end': origins + pd.Timedelta(hours=1),
                             'district': str(district), 'port': str(port)})
        meta['split'] = temporal_split(meta)
        keep = meta.split.ne('purged').to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        arrays.append(windows[good][keep].copy())
        targets.append(y[good][keep])
        metadata.append(meta)
        for split in ('train', 'validation', 'test'):
            audit[split + '_sequences'] = int(meta.split.eq(split).sum())
        audits.append(audit)
    if not arrays or not sum(map(len, arrays)):
        raise ValueError('No valid complete historical sequences')
    meta = pd.concat(metadata, ignore_index=True)
    order = meta.sort_values(['forecast_origin', 'district', 'port']).index.to_numpy()
    return np.concatenate(arrays)[order], np.concatenate(targets)[order], meta.iloc[order].reset_index(drop=True), pd.DataFrame(audits).fillna(0)


def latest_sequence(hourly, district, port, length=24):
    """Latest complete contiguous observed block, never fill gaps or change dates."""
    subset = hourly.loc[hourly.district.eq(district) & hourly.port.eq(port), ['timestamp', 'pm25']].copy()
    subset['timestamp'] = pd.to_datetime(subset.timestamp)
    subset = subset.sort_values('timestamp').dropna(subset=['pm25'])
    if subset.empty:
        raise ValueError('No usable history for selected station')
    segments = subset.timestamp.diff().ne(pd.Timedelta(hours=1)).cumsum()
    candidates = [part for _, part in subset.groupby(segments) if len(part) >= length]
    if not candidates:
        raise ValueError(f'No complete {length}-hour sequence for selected station')
    return candidates[-1].tail(length).reset_index(drop=True)


def load_hourly(root):
    """Read Phase 8 hourly view only after matching its saved content fingerprint."""
    import json
    from ML.model_evaluation import sha256
    root = Path(root)
    path = root / 'REPORTS/ML/phase8/source_audit/eligible_hourly_observations.csv.gz'
    expected = json.loads((root / 'ML/saved_models/phase8/kmeans.metadata.json').read_text())['source_hourly_sha256']
    if sha256(path) != expected:
        raise ValueError('Audited hourly dataset checksum mismatch')
    return pd.read_csv(path, usecols=['timestamp', 'district', 'port', 'pm25'], parse_dates=['timestamp'])


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    X, y, meta, audit = build_sequences(load_hourly(root))
    output = root / 'REPORTS/ML/phase9'
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / 'sequence_continuity_audit.csv', index=False)
    print(audit.to_string(index=False))
    print(meta.groupby('split').size().to_string())
