"""
07_uncertainty.py -- Uncertainty quantification for J1.

CLAUDE.md Section 4.4:
  U1: Bootstrap Ensemble (N=100 RF models -> mean/std/percentiles)
  U2: Quantile Regression Forest (Meinshausen 2006, built on RF leaf samples)
  U3: Conformal Prediction Intervals (split conformal, Vovk et al. 2005)

All three report coverage rates for comparison (Table 7).
"""
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import config


def bootstrap_ensemble(X_train, y_train, X_test, n_bootstrap=None, random_state=None):
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    rng = np.random.default_rng(random_state)
    n = len(X_train)

    preds = np.zeros((n_bootstrap, len(X_test)))
    models = []
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        model = RandomForestRegressor(n_estimators=150, max_depth=20,
                                       random_state=int(rng.integers(0, 1_000_000)), n_jobs=-1)
        model.fit(X_train[idx], y_train[idx])
        preds[i] = model.predict(X_test)
        models.append(model)

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)
    uncertainty_pct = np.where(mean > 0, (upper - lower) / mean * 100, np.nan)

    return {'mean': mean, 'std': std, 'lower': lower, 'upper': upper,
            'uncertainty_pct': uncertainty_pct, 'raw_predictions': preds, 'models': models}


def bootstrap_predict_raster(models, X_grid):
    """Apply an already-fitted bootstrap ensemble (from bootstrap_ensemble's
    'models' list) to new points, e.g. a raster pixel matrix -- avoids
    refitting N models over the (much larger) raster grid."""
    preds = np.stack([m.predict(X_grid) for m in models], axis=0)
    mean = preds.mean(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)
    return {'mean': mean, 'lower': lower, 'upper': upper}


class QuantileRandomForest:
    """Meinshausen (2006) quantile regression forest, built on a standard
    sklearn RandomForestRegressor: at inference, pool the training targets
    that land in the same leaf as the query point across all trees, then
    take empirical quantiles of that pooled distribution."""

    def __init__(self, **rf_params):
        self.rf_params = rf_params
        self.model = RandomForestRegressor(**rf_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.y_train = np.asarray(y_train)
        self._leaf_to_y = []
        train_leaves = self.model.apply(X_train)  # (n_train, n_trees)
        for t in range(train_leaves.shape[1]):
            d = defaultdict(list)
            for i, leaf in enumerate(train_leaves[:, t]):
                d[leaf].append(self.y_train[i])
            self._leaf_to_y.append(d)
        return self

    def predict_quantiles(self, X_test, quantiles=None):
        quantiles = quantiles or config.QUANTILES
        test_leaves = self.model.apply(X_test)  # (n_test, n_trees)
        out = {q: np.zeros(len(X_test)) for q in quantiles}
        for i in range(len(X_test)):
            pooled = []
            for t in range(test_leaves.shape[1]):
                leaf = test_leaves[i, t]
                pooled.extend(self._leaf_to_y[t].get(leaf, []))
            pooled = np.asarray(pooled) if pooled else self.y_train
            for q in quantiles:
                out[q][i] = np.quantile(pooled, q)
        return out


def quantile_regression_forest(X_train, y_train, X_test, quantiles=None):
    quantiles = quantiles or config.QUANTILES
    qrf = QuantileRandomForest(n_estimators=300, max_depth=15,
                                min_samples_leaf=5, random_state=config.RANDOM_STATE, n_jobs=-1)
    qrf.fit(X_train, y_train)
    q_preds = qrf.predict_quantiles(X_test, quantiles)
    lo_q, mid_q, hi_q = min(quantiles), sorted(quantiles)[len(quantiles) // 2], max(quantiles)
    return {'lower': q_preds[lo_q], 'median': q_preds[mid_q], 'upper': q_preds[hi_q],
            'quantiles': q_preds, 'model': qrf}


def conformal_prediction(X_train, y_train, X_test, confidence=None, random_state=None):
    """Split conformal prediction (Vovk, Gammerman & Shafer, 2005).

    1. Hold out 20% of the training set as a calibration set.
    2. Fit a point-prediction model on the remaining 80%.
    3. Nonconformity score = |y_true - y_pred| on calibration set.
    4. Interval half-width = the (1-alpha)-quantile of those scores, with
       finite-sample correction ceil((n_cal+1)(1-alpha))/n_cal.
    5. Apply symmetric interval [pred-w, pred+w] to test predictions.
    """
    confidence = confidence or config.CONFIDENCE_LEVEL
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    alpha = 1 - confidence

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.20, random_state=random_state)

    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(X_fit, y_fit)

    cal_scores = np.abs(y_cal - model.predict(X_cal))
    n_cal = len(cal_scores)
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    half_width = np.quantile(cal_scores, q_level)

    y_pred = model.predict(X_test)
    lower = y_pred - half_width
    upper = y_pred + half_width

    return {'mean': y_pred, 'lower': lower, 'upper': upper,
            'half_width': half_width, 'model': model}


def coverage_rate(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def run(X_train, y_train, X_test, y_test):
    print("  Running Bootstrap Ensemble (U1, N=%d)..." % config.N_BOOTSTRAP)
    boot = bootstrap_ensemble(X_train, y_train, X_test)
    boot_cov = coverage_rate(y_test, boot['lower'], boot['upper'])
    print(f"    Bootstrap 90% PI coverage: {boot_cov:.3f}")

    print("  Running Quantile Regression Forest (U2)...")
    qrf = quantile_regression_forest(X_train, y_train, X_test)
    qrf_cov = coverage_rate(y_test, qrf['lower'], qrf['upper'])
    print(f"    QRF 90% PI coverage: {qrf_cov:.3f}")

    print("  Running Conformal Prediction (U3)...")
    conf = conformal_prediction(X_train, y_train, X_test)
    conf_cov = coverage_rate(y_test, conf['lower'], conf['upper'])
    print(f"    Conformal 90% PI coverage: {conf_cov:.3f}")

    return {
        'bootstrap': {**boot, 'coverage': boot_cov},
        'qrf': {**qrf, 'coverage': qrf_cov},
        'conformal': {**conf, 'coverage': conf_cov},
    }


if __name__ == "__main__":
    pass
