"""Reusable frozen LSTM inference for historical, manual and sample adapters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ML.sequence_builder import FEATURES, encode_observations
from ML.model_evaluation import sha256


def validate_sequence(records, metadata):
    """Validate exact model schema, chronological hourly alignment and ranges.

    Only PM2.5 is entered; six calendar features are derived deterministically.
    No missing values, interpolation, duplicate times or unknown input columns.
    """
    if not isinstance(records, list) or len(records) != metadata['sequence_length']:
        raise ValueError(f"Exactly {metadata['sequence_length']} hourly observations are required")
    allowed = {'timestamp', *metadata['observation_features']}
    if any(not isinstance(row, dict) or set(row) != allowed for row in records):
        raise ValueError('Each observation must contain exactly timestamp and the required pollutant fields')
    try:
        times = pd.DatetimeIndex(pd.to_datetime([row['timestamp'] for row in records], errors='raise'))
    except (ValueError, TypeError):
        raise ValueError('Invalid or mixed-format timestamps') from None
    if times.tz is not None:
        times = times.tz_convert('Asia/Kolkata').tz_localize(None)
    if times.hasnans or not times.equals(times.floor('h')) or not times.is_unique:
        raise ValueError('Timestamps must be unique whole-hour interval starts')
    if not times.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=metadata['sampling_interval_minutes'])).all():
        raise ValueError('Observations must be increasing, consecutive hourly intervals; gaps are not allowed')
    values = []
    for row in records:
        value = row['pm25']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('PM2.5 must be a finite number, not blank/text/boolean')
        if not metadata['input_validation']['pm25']['minimum'] <= value <= metadata['input_validation']['pm25']['maximum']:
            raise ValueError('PM2.5 outside supported training range; this is a model-support limit, not a safety threshold')
        values.append(value)
    return times, encode_observations(times, values)


class LSTMForecaster:
    """Load tensor-only model state and frozen scaler once; infer without fitting."""
    def __init__(self, model_directory):
        import torch
        from ML.lstm_model import PollutantLSTM
        directory = Path(model_directory)
        self.metadata = json.loads((directory / 'lstm_metadata.json').read_text())
        self.scaler = json.loads((directory / 'lstm_scaler.json').read_text())
        if self.metadata['features'] != FEATURES or self.metadata['forecast_horizon_hours'] != 1:
            raise ValueError('Unsupported saved model feature/horizon contract')
        if sha256(directory / 'lstm_model.pt') != self.metadata['model_sha256'] or sha256(directory / 'lstm_scaler.json') != self.metadata['scaler_sha256']:
            raise ValueError('Model/scaler checksum mismatch')
        torch.set_num_threads(2)
        self.model = PollutantLSTM(features=len(FEATURES), hidden=self.metadata['architecture']['hidden_units'])
        self.model.load_state_dict(torch.load(directory / 'lstm_model.pt', map_location='cpu', weights_only=True))
        self.model.eval()

    def forecast(self, records, horizon=1):
        import torch
        if type(horizon) is not int or horizon != 1:
            raise ValueError('Only the trained and evaluated next-1-hour horizon is supported')
        times, X = validate_sequence(records, self.metadata)
        scaled = ((X - np.asarray(self.scaler['feature_mean'])) / np.asarray(self.scaler['feature_std'])).astype('float32')
        with torch.inference_mode():
            normalized = self.model(torch.from_numpy(scaled[None])).item()
        value = max(0., normalized*self.scaler['target_std'] + self.scaler['target_mean'])
        if not np.isfinite(value):
            raise ValueError('Model returned a nonfinite prediction')
        origin = times[-1] + pd.Timedelta(hours=2)
        return {'target': 'pm25', 'predicted_value': value, 'units': self.metadata['units'],
                'forecast_origin': origin.isoformat(), 'target_start': origin.isoformat(),
                'target_end': (origin + pd.Timedelta(hours=1)).isoformat(), 'forecast_horizon_hours': 1,
                'latest_observation_start': times[-1].isoformat(),
                'latest_observation_end': (times[-1] + pd.Timedelta(hours=1)).isoformat(),
                'availability_gap_hours': 1, 'model_id': self.metadata['model_id']}
