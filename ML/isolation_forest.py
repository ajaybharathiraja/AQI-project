"""Independent station-specific unusual-pollution detector; no DS imports.

Detects unusual observed hourly levels/combinations, not future events or proven
health risks. High-tail overlap is a human-review signal, not ground truth.
"""
import numpy as np
import pandas as pd
from ML.unsupervised_data import POLLUTANTS, validate_hourly


class PollutionAnomalies:
    """Historical station forests with fixed 1% reference contamination policy.

    Sensors require >=50% coverage and >=500 valid reference hours by default;
    each model requires >=2 varying pollutants and >=500 complete-case hours.
    Missing observations are explicitly NOT scored, rather than imputed spikes.
    """
    def __init__(self, reference_end='2025-01-01', contamination=.01,
                 minimum_reference_rows=500, minimum_coverage=.5, seed=42, n_jobs=2):
        if not 0 < contamination <= .5 or not 0 < minimum_coverage <= 1 or minimum_reference_rows < 10:
            raise ValueError('Invalid contamination, coverage or minimum reference size')
        self.reference_end = reference_end
        self.contamination = contamination
        self.minimum_reference_rows = minimum_reference_rows
        self.minimum_coverage = minimum_coverage
        self.seed, self.n_jobs = seed, n_jobs

    def fit(self, hourly):
        from sklearn.ensemble import IsolationForest
        available = [col for col in POLLUTANTS if col in hourly]
        data = validate_hourly(hourly, available)
        # Fit complete intervals available by the reference cutoff, never later.
        reference = data.loc[(data.timestamp + pd.Timedelta(hours=1)).le(pd.Timestamp(self.reference_end))]
        self.models_, self.audit_ = {}, []
        for key, group in reference.groupby(['district', 'port'], observed=True):
            features = [col for col in available if group[col].count() >= self.minimum_reference_rows
                        and group[col].notna().mean() >= self.minimum_coverage and group[col].nunique() > 1]
            complete = group.dropna(subset=features)
            audit = {'district': str(key[0]), 'port': str(key[1]), 'reference_hour_bins': len(group),
                     'features': ','.join(features), 'complete_reference_hours': len(complete)}
            if len(features) < 2 or len(complete) < self.minimum_reference_rows:
                audit['status'] = 'insufficient_reference'
                self.audit_.append(audit)
                continue
            values = complete[features].to_numpy(dtype=float)
            model = IsolationForest(n_estimators=150, max_samples=min(256, len(complete)),
                                    contamination=self.contamination, random_state=self.seed,
                                    n_jobs=self.n_jobs).fit(np.log1p(values))
            q99 = complete[features].quantile(.99).to_dict()
            self.models_[tuple(map(str, key))] = {'model': model, 'features': features,
                                                'q99': q99, 'reference_rows': len(complete)}
            audit['status'] = 'fitted'
            self.audit_.append(audit)
        if not self.models_:
            raise ValueError('No station has enough complete historical pollutant features')
        return self

    def analyze(self, hourly):
        """Return identity, original pollutant values, scores and explicit status.

        anomaly_score = -score_samples: larger means more unusual, NOT probability.
        decision_function <0 marks anomaly. high_value_review additionally needs
        some modeled pollutant above its station's historical 99th percentile.
        review_priority is max(value/q99 - 1, 0); no health/AQI threshold is used.
        """
        if not hasattr(self, 'models_'):
            raise ValueError('Fit reference before anomaly scoring')
        available = [col for col in POLLUTANTS if col in hourly]
        data = validate_hourly(hourly, available)
        result = data[['timestamp', 'district', 'port'] + available].copy()
        result['observation_end'] = result.timestamp + pd.Timedelta(hours=1)
        result['period'] = np.where(result.observation_end.le(pd.Timestamp(self.reference_end)), 'reference', 'monitoring')
        result['anomaly_score'] = np.nan
        result['decision_function'] = np.nan
        result['status'] = 'not_scored_no_reference_model'
        result['anomaly'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
        result['modeled_pollutants'] = ''
        result['high_tail_pollutants'] = ''
        result['high_value_review'] = False
        result['review_priority'] = np.nan
        for key, group in result.groupby(['district', 'port'], observed=True):
            record = self.models_.get(tuple(map(str, key)))
            if record is None:
                continue
            features = record['features']
            result.loc[group.index, 'modeled_pollutants'] = ','.join(features)
            result.loc[group.index, 'status'] = 'not_scored_missing_pollutants'
            if any(col not in group for col in features):
                continue
            complete = group.dropna(subset=features)
            if complete.empty:
                continue
            values = complete[features]
            scores = record['model'].score_samples(np.log1p(values.to_numpy(dtype=float)))
            decision = scores - record['model'].offset_
            anomaly = decision < 0
            exceed = values.gt(pd.Series(record['q99']))
            # A zero percentile has no meaningful ratio; do not divide by epsilon
            # and invent immense priorities. Values may still be flagged for review.
            denominators = pd.Series(record['q99']).replace(0, np.nan)
            priority = (values.div(denominators) - 1).clip(lower=0).max(axis=1).fillna(0)
            index = complete.index
            result.loc[index, 'anomaly_score'] = -scores
            result.loc[index, 'decision_function'] = decision
            result.loc[index, 'anomaly'] = anomaly
            result.loc[index, 'status'] = np.where(anomaly, 'anomaly', 'normal')
            result.loc[index, 'high_value_review'] = anomaly & exceed.any(axis=1).to_numpy()
            result.loc[index, 'review_priority'] = priority.to_numpy()
            result.loc[index, 'high_tail_pollutants'] = [','.join(col for col in features if row[col]) for _, row in exceed.iterrows()]
        return result
