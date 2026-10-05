"""CPU histogram gradient boosting; independent of custom data structures."""


def create_model(n_jobs=2):
    """Return fixed XGBoost candidate; no test-set early stopping or tuning."""
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.05,
                        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=1.0, objective='reg:squarederror',
                        tree_method='hist', device='cpu', n_jobs=n_jobs,
                        random_state=42, verbosity=0)
