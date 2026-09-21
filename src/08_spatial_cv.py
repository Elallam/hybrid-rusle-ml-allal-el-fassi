"""
08_spatial_cv.py -- Spatial Leave-One-Out CV (SLOOCV) + spatial block CV.

CLAUDE.md Section 6. This is the methodological rigor that separates J1
from C2: standard random K-fold CV lets spatially autocorrelated neighbours
leak between train/test, inflating metrics (Roberts et al., 2017; Valavi
et al., 2019). SLOOCV enforces a geographic exclusion buffer around each
test point; spatial block CV (KMeans blocks) gives a cheaper 5-fold
approximation for the deep learning models.
"""
import os
import importlib.util
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor

import config

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SRC_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SpatialLOOCV:
    """Spatial Leave-One-Out Cross-Validation (Roberts et al., 2017;
    Valavi et al., 2019). For each test point, all training points within
    `buffer_distance` metres are excluded before refitting."""

    def __init__(self, buffer_distance=None, max_points=None, random_state=None):
        self.buffer_distance = buffer_distance or config.BUFFER_DIST
        self.max_points = max_points or config.SLOOCV_MAX_POINTS
        self.random_state = random_state if random_state is not None else config.RANDOM_STATE

    def split(self, coords):
        coords = np.asarray(coords)
        n = len(coords)
        idx_pool = np.arange(n)
        if self.max_points and n > self.max_points:
            rng = np.random.default_rng(self.random_state)
            test_points = np.sort(rng.choice(n, self.max_points, replace=False))
        else:
            test_points = idx_pool
        d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
        for i in test_points:
            train_mask = d[i] > self.buffer_distance
            yield idx_pool[train_mask], np.array([i])

    def cross_val_score(self, model_factory, X, y, coords):
        y_true_all, y_pred_all = [], []
        for train_idx, test_idx in self.split(coords):
            if len(train_idx) < 20:
                continue
            m = model_factory()
            m.fit(X[train_idx], y[train_idx])
            y_pred_all.append(m.predict(X[test_idx]))
            y_true_all.append(y[test_idx])
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        return {'R2': r2, 'RMSE': rmse, 'n_test': len(y_true_all)}


def spatial_block_folds(coords, n_folds=None, random_state=None):
    n_folds = n_folds or config.N_FOLDS_BLOCK
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    coords = np.asarray(coords)
    km = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    block_id = km.fit_predict(coords)
    idx = np.arange(len(coords))
    return [(idx[block_id != b], idx[block_id == b]) for b in range(n_folds)]


def standard_kfold_folds(n_samples, n_folds=None, random_state=None):
    n_folds = n_folds or config.N_FOLDS_BLOCK
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def _ml_factories(features):
    ml04 = _load_sibling("ml04_cv", "04_ml_models.py")
    monotone = ml04.build_monotone_constraints(features)
    return {
        'Random Forest': lambda: RandomForestRegressor(**config.RF_PARAMS),
        'XGBoost': lambda: xgb.XGBRegressor(**config.XGB_PARAMS, monotone_constraints=monotone),
        'Gradient Boosting': lambda: GradientBoostingRegressor(**config.GB_PARAMS),
        'CatBoost': lambda: CatBoostRegressor(**config.CATBOOST_PARAMS),
    }


def _ml_factories_single_threaded(features):
    """Same base learners as _ml_factories but pinned to n_jobs=1. Nesting
    StackingRegressor's own n_jobs=-1 parallel base-learner fitting inside
    RF/XGB estimators that ALSO try to spawn n_jobs=-1 worker pools crashes
    joblib's loky backend on Windows (TerminatedWorkerError) -- single
    threading is essential wherever a stacking model is repeatedly refit in
    a CV loop on data this small, where parallelism buys nothing anyway."""
    ml04 = _load_sibling("ml04_cv_st", "04_ml_models.py")
    monotone = ml04.build_monotone_constraints(features)
    rf_params = {**config.RF_PARAMS, 'n_jobs': 1}
    xgb_params = {**config.XGB_PARAMS, 'n_jobs': 1}
    return {
        'Random Forest': lambda: RandomForestRegressor(**rf_params),
        'XGBoost': lambda: xgb.XGBRegressor(**xgb_params, monotone_constraints=monotone),
        'Gradient Boosting': lambda: GradientBoostingRegressor(**config.GB_PARAMS),
        'CatBoost': lambda: CatBoostRegressor(**{**config.CATBOOST_PARAMS, 'thread_count': 1}),
    }


def run_sloocv(features, X, y, coords, model_names=('Random Forest', 'XGBoost')):
    """SLOOCV is expensive -> restricted to RF & XGBoost per CLAUDE.md Section 6."""
    factories = _ml_factories(features)
    sloocv = SpatialLOOCV()
    out = {}
    for name in model_names:
        print(f"  SLOOCV for {name} (buffer={sloocv.buffer_distance}m, "
              f"max_points={sloocv.max_points})...")
        out[name] = sloocv.cross_val_score(factories[name], X, y, coords)
        print(f"    {name}: SLOOCV R2={out[name]['R2']:.4f}, RMSE={out[name]['RMSE']:.3f} "
              f"(n={out[name]['n_test']})")
    return out


def _fold_metrics(y_true, y_pred):
    return r2_score(y_true, y_pred)


def run_block_cv(features, X_std, y, coords, n_folds=None):
    """5-fold spatial block CV vs standard 5-fold CV, for the 4 base ML
    models + Stacking Ensemble (fast enough to run every fold)."""
    n_folds = n_folds or config.N_FOLDS_BLOCK
    base_factories = _ml_factories(features)
    # Single-threaded variants ONLY for the base learners nested inside the
    # stacking model -- see _ml_factories_single_threaded's docstring.
    base_factories_st = _ml_factories_single_threaded(features)

    def make_stack():
        # NOTE: closes over base_factories_st (4 base learners only), never
        # over `factories` below -- closing over `factories` would recurse
        # forever once 'Stacking Ensemble' -> make_stack is inserted into it.
        base = {k: f() for k, f in base_factories_st.items()}
        estimators = [('rf', base['Random Forest']), ('xgb', base['XGBoost']),
                      ('gb', base['Gradient Boosting']), ('cb', base['CatBoost'])]
        from sklearn.ensemble import StackingRegressor
        return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0),
                                  cv=3, passthrough=True, n_jobs=1)

    factories = dict(base_factories)
    factories['Stacking Ensemble'] = make_stack

    spatial_folds = spatial_block_folds(coords, n_folds)
    standard_folds = standard_kfold_folds(len(y), n_folds)

    rows = []
    for name, factory in factories.items():
        spatial_r2, standard_r2 = [], []
        for train_idx, test_idx in spatial_folds:
            m = factory()
            m.fit(X_std[train_idx], y[train_idx])
            spatial_r2.append(_fold_metrics(y[test_idx], m.predict(X_std[test_idx])))
        for train_idx, test_idx in standard_folds:
            m = factory()
            m.fit(X_std[train_idx], y[train_idx])
            standard_r2.append(_fold_metrics(y[test_idx], m.predict(X_std[test_idx])))

        row = {'Model': name}
        for i, r2 in enumerate(spatial_r2, 1):
            row[f'Fold{i}'] = r2
        row['Mean_R2_spatial'] = float(np.mean(spatial_r2))
        row['Std_R2_spatial'] = float(np.std(spatial_r2))
        row['Mean_R2_standard'] = float(np.mean(standard_r2))
        row['Std_R2_standard'] = float(np.std(standard_r2))
        row['Difference'] = row['Mean_R2_standard'] - row['Mean_R2_spatial']
        rows.append(row)
        print(f"  Block CV {name}: spatial R2={row['Mean_R2_spatial']:.4f}+/-{row['Std_R2_spatial']:.4f}, "
              f"standard R2={row['Mean_R2_standard']:.4f}+/-{row['Std_R2_standard']:.4f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pass
