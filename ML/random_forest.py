"""Random Forest regression factory; no custom data-structure dependencies."""


def create_model(n_jobs=2):
    """Fixed, bounded candidate; bootstrap sampling is confined to TRAIN rows.

    No shuffled validation or out-of-bag temporal score is used. Predictors and
    targets are supplied by model_evaluation, never loaded from source here.
    """
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=80, max_depth=14,
                                 min_samples_leaf=5, max_features=0.7,
                                 max_samples=0.8, bootstrap=True, oob_score=False,
                                 n_jobs=n_jobs, random_state=42)
