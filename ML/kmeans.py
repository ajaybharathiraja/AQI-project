"""Station-day pollution profiles, NOT geographic hotspot identification.

Use PollutionClusters.fit(daily) on a historical reference, then assign(daily).
No custom data structures, location encodings, coordinates or labels enter X.
"""
import numpy as np
import pandas as pd
from ML.unsupervised_data import validate_hourly

FEATURES = ['pm25', 'pm10', 'no2']


def station_days(hourly, minimum_hours=18):
    """Daily means from >=18 jointly valid hourly observations (same hours).

    Return all observed station-days with counts and eligibility. Ineligible
    daily means remain descriptive only and must not receive cluster labels.
    """
    if type(minimum_hours) is not int or not 1 <= minimum_hours <= 24:
        raise ValueError('minimum_hours must be an integer in 1..24')
    frame = validate_hourly(hourly, FEATURES)
    frame['date'] = frame.timestamp.dt.floor('D')
    keys = ['district', 'port', 'date']
    base = frame.groupby(keys, observed=True).size().rename('observed_hour_bins').to_frame()
    complete = frame.dropna(subset=FEATURES)
    daily = complete.groupby(keys, observed=True)[FEATURES].mean()
    daily['joint_valid_hours'] = complete.groupby(keys, observed=True).size()
    result = base.join(daily).reset_index()
    result['joint_valid_hours'] = result.joint_valid_hours.fillna(0).astype(int)
    result['eligible'] = result.joint_valid_hours.ge(minimum_hours)
    return result.sort_values(['date', 'district', 'port']).reset_index(drop=True)


class PollutionClusters:
    """Log1p + training-reference StandardScaler + silhouette-selected K-Means.

    Features have equal standardized weight; no raw-unit distance mixing.
    Selection uses reference days only, not later monitoring days. IDs have no
    severity ordering. K-Means is exploratory, not a geographic density test.
    """
    def __init__(self, reference_end='2025-01-01', candidates=(2, 3, 4, 5), seed=42):
        self.reference_end = reference_end
        self.candidates = candidates
        self.seed = seed

    def fit(self, daily):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        reference = daily.loc[daily.eligible & (daily.date + pd.Timedelta(days=1)).le(pd.Timestamp(self.reference_end))]
        if len(reference) < 100:
            raise ValueError('Need at least 100 eligible reference station-days')
        if not self.candidates or any(type(k) is not int or k < 2 or k >= len(reference) for k in self.candidates):
            raise ValueError('Candidate k must be integer >=2 and smaller than reference rows')
        values = reference[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Eligible daily pollutant features must be finite/nonnegative')
        self.scaler_ = StandardScaler().fit(np.log1p(values))
        X = self.scaler_.transform(np.log1p(values))
        if (self.scaler_.var_ <= 0).any():
            raise ValueError('Clustering requires variable reference features')
        rng = np.random.default_rng(self.seed)
        sample = rng.choice(len(X), min(2000, len(X)), replace=False)
        self.selection_ = []
        fitted = {}
        for k in self.candidates:
            model = KMeans(n_clusters=k, n_init=10, random_state=self.seed, max_iter=300).fit(X)
            labels = model.labels_
            minimum = int(np.bincount(labels, minlength=k).min())
            sample_labels = labels[sample]
            silhouette = float(silhouette_score(X[sample], sample_labels)) if 1 < len(np.unique(sample_labels)) < len(sample) else None
            supported = minimum >= max(20, int(np.ceil(.01 * len(X)))) and silhouette is not None
            self.selection_.append({'k': k, 'silhouette_reference_sample': silhouette,
                                    'inertia': float(model.inertia_), 'minimum_reference_cluster_rows': minimum,
                                    'supported': supported, 'silhouette_sample_rows': len(sample)})
            fitted[k] = model
        eligible = [row for row in self.selection_ if row['supported']]
        if not eligible:
            raise ValueError('No candidate has adequate reference cluster support')
        selected = min(eligible, key=lambda row: (-row['silhouette_reference_sample'], row['k']))
        self.model_ = fitted[selected['k']]
        self.reference_rows_ = len(reference)
        self.reference_means_ = reference[FEATURES].mean().to_dict()
        return self

    def assign(self, daily):
        """Label only complete eligible days; missing/ineligible days get NA."""
        if not hasattr(self, 'model_'):
            raise ValueError('Fit historical reference before assigning clusters')
        result = daily.copy()
        result['cluster'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
        eligible = result.eligible
        if eligible.any():
            values = result.loc[eligible, FEATURES].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Eligible measurements must be finite and nonnegative')
            X = self.scaler_.transform(np.log1p(values))
            result.loc[eligible, 'cluster'] = self.model_.predict(X)
        result['period'] = np.where((result.date + pd.Timedelta(days=1)).le(pd.Timestamp(self.reference_end)), 'reference', 'monitoring')
        return result

    def centers(self):
        """Inverse-log transformed centers are NOT arithmetic pollution means."""
        frame = pd.DataFrame(np.expm1(self.scaler_.inverse_transform(self.model_.cluster_centers_)), columns=FEATURES)
        frame.insert(0, 'cluster', range(len(frame)))
        return frame
