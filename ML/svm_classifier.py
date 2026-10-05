"""Separate supervised SVM classification; no custom data structures or AQI thresholds.

AQI labels come from the existing rule-based engine. Scaling is fitted only by
Pipeline.fit on the chronological training subset. No probability is fabricated.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from AQI.aqi_category import aqi_category
from AQI.aqi_breakpoints import INDEX_BANDS


def target_classes():
    """Exact category names from the engine, without redefining its thresholds."""
    return [aqi_category(upper) for _, upper in INDEX_BANDS] + [aqi_category(INDEX_BANDS[-1][1] + 1)]


def create_model(kernel='linear', C=1.0, gamma='scale'):
    if kernel == 'linear':
        classifier = LinearSVC(C=C, class_weight='balanced', dual=False,
                               max_iter=20000, tol=1e-4, random_state=42)
    elif kernel == 'rbf':
        classifier = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced',
                         cache_size=128, probability=False, random_state=42)
    else:
        raise ValueError('Only the predeclared linear and RBF candidates are supported')
    return Pipeline([('scaler', StandardScaler()), ('svm', classifier)])


def classification_metrics(actual, predicted):
    labels = target_classes()
    if not len(actual) or len(actual) != len(predicted) or not set(actual).issubset(labels) or not set(predicted).issubset(labels):
        raise ValueError('Invalid category labels or unmatched/empty vectors')
    p, r, f, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
    weighted = float(np.average(f, weights=support))
    return {'accuracy': float(accuracy_score(actual, predicted)), 'macro_precision': float(p.mean()),
            'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()), 'weighted_f1': weighted,
            'per_class': [{'category': name, 'precision': float(p[i]), 'recall': float(r[i]),
                           'f1': float(f[i]), 'support': int(support[i])} for i, name in enumerate(labels)],
            'confusion_matrix': confusion_matrix(actual, predicted, labels=labels).tolist(),
            'averaging': 'Macro across all six engine categories, including absent classes with zero_division=0; weighted by actual support'}
