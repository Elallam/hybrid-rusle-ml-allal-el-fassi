"""
04_ml_models.py -- Base ML models for J1 (retrained from C2 + CatBoost, new).

CLAUDE.md Section 4.1:
  MODEL 1: Random Forest      (+ bootstrap predict_interval, see 07_uncertainty.py)
  MODEL 2: XGBoost             (+ monotone constraints for physical consistency)
  MODEL 3: Gradient Boosting
  MODEL 4: CatBoost            (new in J1 -- handles categorical features natively)
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import xgboost as xgb
from catboost import CatBoostRegressor

import config
from utils import confirm_saved


def build_monotone_constraints(features):
    cons = []
    for f in features:
        if f in config.XGB_MONOTONE_POSITIVE:
            cons.append(1)
        elif f in config.XGB_MONOTONE_NEGATIVE:
            cons.append(-1)
        else:
            cons.append(0)
    return "(" + ",".join(str(c) for c in cons) + ")"


def hyperparameter_search(features, X_train, y_train):
    """Lightweight RandomizedSearchCV for RF & XGB -> Table 3.
    GB/CatBoost use literature-informed fixed configs (config.py) -- documented
    as such rather than fabricated as "optimized"."""
    rows = []
    cv = KFold(3, shuffle=True, random_state=config.RANDOM_STATE)

    rf_grid = {'n_estimators': [300, 500, 700], 'max_depth': [None, 15, 25],
               'min_samples_leaf': [1, 2, 4]}
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
        rf_grid, n_iter=6, cv=cv, scoring='r2', random_state=config.RANDOM_STATE, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    for p, v in rf_search.best_params_.items():
        rows.append({'Model': 'Random Forest', 'Parameter': p,
                      'Search_Range': str(rf_grid[p]), 'Best_Value': v,
                      'CV_R2': rf_search.best_score_})

    xgb_grid = {'max_depth': [4, 6, 8], 'learning_rate': [0.01, 0.03, 0.05],
                'n_estimators': [400, 600, 800]}
    xgb_gpu_kwargs = {'device': 'cuda', 'tree_method': 'hist'} if config.USE_GPU else {'n_jobs': -1}
    # NOTE: RandomizedSearchCV itself must stay n_jobs=1 on GPU -- there is
    # only one GPU, and sklearn's own parallel CV workers would each try to
    # grab it at once (contention/CUDA context errors on a single device).
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=config.RANDOM_STATE, **xgb_gpu_kwargs),
        xgb_grid, n_iter=6, cv=cv, scoring='r2', random_state=config.RANDOM_STATE,
        n_jobs=(1 if config.USE_GPU else -1))
    xgb_search.fit(X_train, y_train)
    for p, v in xgb_search.best_params_.items():
        rows.append({'Model': 'XGBoost', 'Parameter': p,
                      'Search_Range': str(xgb_grid[p]), 'Best_Value': v,
                      'CV_R2': xgb_search.best_score_})

    for p, v in config.GB_PARAMS.items():
        rows.append({'Model': 'Gradient Boosting', 'Parameter': p,
                      'Search_Range': 'fixed (literature-informed)', 'Best_Value': v,
                      'CV_R2': np.nan})
    for p, v in config.CATBOOST_PARAMS.items():
        rows.append({'Model': 'CatBoost', 'Parameter': p,
                      'Search_Range': 'fixed (literature-informed)', 'Best_Value': v,
                      'CV_R2': np.nan})

    return pd.DataFrame(rows), rf_search.best_params_, xgb_search.best_params_


def train_all(features, X_train, y_train, X_test, tune=True):
    results = {}

    rf_params = dict(config.RF_PARAMS)
    xgb_params = dict(config.XGB_PARAMS)

    if tune:
        tuning_table, rf_best, xgb_best = hyperparameter_search(features, X_train, y_train)
        rf_params.update(rf_best)
        xgb_params.update(xgb_best)
    else:
        # Quick/smoke-test mode: skip the RandomizedSearchCV, but still return
        # a valid Table 3 documenting the fixed config values actually used.
        rows = []
        for name, params in (('Random Forest', config.RF_PARAMS), ('XGBoost', config.XGB_PARAMS),
                              ('Gradient Boosting', config.GB_PARAMS),
                              ('CatBoost', config.CATBOOST_PARAMS)):
            for p, v in params.items():
                rows.append({'Model': name, 'Parameter': p,
                              'Search_Range': 'fixed (tuning skipped this run)',
                              'Best_Value': v, 'CV_R2': np.nan})
        tuning_table = pd.DataFrame(rows)

    # -- Random Forest -------------------------------------------------------
    t0 = time.time()
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    results['Random Forest'] = {'model': rf, 'y_pred': rf.predict(X_test),
                                 'train_time': time.time() - t0}

    # -- XGBoost (monotone constraints for physical consistency) ------------
    t0 = time.time()
    monotone = build_monotone_constraints(features)
    xgb_model = xgb.XGBRegressor(**xgb_params, monotone_constraints=monotone)
    xgb_model.fit(X_train, y_train)
    results['XGBoost'] = {'model': xgb_model, 'y_pred': xgb_model.predict(X_test),
                           'train_time': time.time() - t0}

    # -- Gradient Boosting ----------------------------------------------------
    t0 = time.time()
    gb = GradientBoostingRegressor(**config.GB_PARAMS)
    gb.fit(X_train, y_train)
    results['Gradient Boosting'] = {'model': gb, 'y_pred': gb.predict(X_test),
                                     'train_time': time.time() - t0}

    # -- CatBoost (new in J1) --------------------------------------------------
    t0 = time.time()
    cb = CatBoostRegressor(**config.CATBOOST_PARAMS)
    cb.fit(X_train, y_train)
    results['CatBoost'] = {'model': cb, 'y_pred': cb.predict(X_test),
                            'train_time': time.time() - t0}

    for name, r in results.items():
        print(f"  {name:<20} trained in {r['train_time']:.1f}s")

    return results, tuning_table


def save_models(results):
    joblib.dump(results['Random Forest']['model'],
                os.path.join(config.MODELS_DIR, "rf_final.pkl"))
    confirm_saved(os.path.join(config.MODELS_DIR, "rf_final.pkl"))
    results['XGBoost']['model'].save_model(os.path.join(config.MODELS_DIR, "xgb_final.json"))
    confirm_saved(os.path.join(config.MODELS_DIR, "xgb_final.json"))
    joblib.dump(results['Gradient Boosting']['model'],
                os.path.join(config.MODELS_DIR, "gb_final.pkl"))
    confirm_saved(os.path.join(config.MODELS_DIR, "gb_final.pkl"))
    results['CatBoost']['model'].save_model(
        os.path.join(config.MODELS_DIR, "catboost_final", "model.cbm"))
    confirm_saved(os.path.join(config.MODELS_DIR, "catboost_final", "model.cbm"))


if __name__ == "__main__":
    pass
