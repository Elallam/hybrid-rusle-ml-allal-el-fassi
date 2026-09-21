"""
05_stacking_ensemble.py -- Stacked ensemble: the MAIN NOVELTY of J1.

CLAUDE.md Section 4.2.
Level 0 (base learners) : Random Forest, XGBoost, Gradient Boosting, CatBoost
Level 1 (meta learner)  : Ridge Regression (alpha=1.0), passthrough=True

Also runs a leave-one-base-out ablation study, reported in the paper as
evidence for why each base learner contributes to the ensemble.
"""
import os
import time
import importlib.util
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

import config
from utils import confirm_saved


def _fresh_base_learners(features):
    # Reuse 04_ml_models.py's monotone-constraint builder (identical physical
    # constraints as the standalone XGBoost model) without re-running its
    # hyperparameter search -- import the sibling module via file path since
    # a filename starting with a digit isn't a valid `import` identifier.
    spec = importlib.util.spec_from_file_location(
        "ml04_stack", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_ml_models.py"))
    ml04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml04)
    monotone = ml04.build_monotone_constraints(features)

    return {
        'rf': RandomForestRegressor(**config.RF_PARAMS),
        'xgb': xgb.XGBRegressor(**config.XGB_PARAMS, monotone_constraints=monotone),
        'gb': GradientBoostingRegressor(**config.GB_PARAMS),
        'cb': CatBoostRegressor(**config.CATBOOST_PARAMS),
    }


def build_stacking(features, X_train, y_train, X_test, y_test):
    base = _fresh_base_learners(features)
    estimators = [('rf', base['rf']), ('xgb', base['xgb']),
                  ('gb', base['gb']), ('cb', base['cb'])]

    meta = Ridge(alpha=1.0) if config.STACKING_META == 'ridge' else LinearRegression()

    t0 = time.time()
    # n_jobs=1 at the StackingRegressor level (not -1): RF/XGB base learners
    # already parallelize internally (config.RF_PARAMS/XGB_PARAMS n_jobs=-1),
    # and nesting StackingRegressor's own worker-process pool around base
    # learners that each spawn their own pool crashes joblib's loky backend
    # on Windows (TerminatedWorkerError) -- fit the 4 base learners
    # sequentially, each still using all cores internally.
    stack = StackingRegressor(estimators=estimators, final_estimator=meta,
                               cv=5, passthrough=True, n_jobs=1)
    stack.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = stack.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"  Stacking Ensemble trained in {train_time:.1f}s, test R2={r2:.4f}")
    print(f"  Meta-learner (Ridge) coefficients per base learner: "
          f"{dict(zip(['rf', 'xgb', 'gb', 'cb'], stack.final_estimator_.coef_[:4]))}")

    return {'model': stack, 'y_pred': y_pred, 'train_time': train_time}


def ablation_study(features, X_train, y_train, X_test, y_test, full_r2):
    """Leave-one-base-out: retrain stacking with each base learner removed in turn."""
    base_names = ['rf', 'xgb', 'gb', 'cb']
    rows = [{'Configuration': 'Full stack (RF+XGB+GB+CatBoost)', 'R2': full_r2, 'Delta_R2': 0.0}]

    for drop in base_names:
        base = _fresh_base_learners(features)
        estimators = [(n, base[n]) for n in base_names if n != drop]
        meta = Ridge(alpha=1.0) if config.STACKING_META == 'ridge' else LinearRegression()
        stack = StackingRegressor(estimators=estimators, final_estimator=meta,
                                   cv=5, passthrough=True, n_jobs=1)  # see build_stacking() note
        stack.fit(X_train, y_train)
        r2 = r2_score(y_test, stack.predict(X_test))
        rows.append({'Configuration': f"Without {drop.upper()}", 'R2': r2,
                      'Delta_R2': r2 - full_r2})
        print(f"  Ablation (without {drop}): R2={r2:.4f} (delta={r2 - full_r2:+.4f})")

    return pd.DataFrame(rows)


def save_model(stack_result):
    path = os.path.join(config.MODELS_DIR, "stacking_ensemble.pkl")
    joblib.dump(stack_result['model'], path)
    confirm_saved(path)


if __name__ == "__main__":
    pass
