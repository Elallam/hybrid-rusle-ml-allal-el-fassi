"""
09_evaluation.py -- Full evaluation metric suite for J1 (Catena publication).

CLAUDE.md Section 7. Regression (R2, Adj_R2, RMSE, NRMSE, MAE, MBE, NSE,
KGE, PBIAS, RMSE_STD, IOA), classification (OA, Kappa, F1_macro on erosion
classes), uncertainty (Coverage, MPIW, PICP) and spatial (Moran's I of
residuals) metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, cohen_kappa_score, f1_score)
from sklearn.neighbors import NearestNeighbors

import config


def morans_i(residuals, coords, k=8):
    """Moran's I of model residuals over a k-nearest-neighbour spatial
    weight matrix (row-standardized). Target: I ~ 0 (no residual spatial
    pattern left uncaptured by the model)."""
    residuals = np.asarray(residuals, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n = len(residuals)
    k = min(k, n - 1)
    if n < 3:
        return np.nan, np.nan

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, neighbor_idx = nn.kneighbors(coords)
    neighbor_idx = neighbor_idx[:, 1:]  # drop self

    x = residuals - residuals.mean()
    num = 0.0
    S0 = 0.0
    for i in range(n):
        w = 1.0 / k
        num += w * x[i] * x[neighbor_idx[i]].sum()
        S0 += k * w
    denom = np.sum(x ** 2)
    I = (n / S0) * (num / denom) if denom > 0 else np.nan

    # Expected value and simple z-approximation under randomization null
    E_I = -1.0 / (n - 1)
    var_I = (1.0 / (S0 ** 2 * (n - 1))) * (n ** 2 * k - n * k ** 2) if n > 1 else np.nan
    z = (I - E_I) / np.sqrt(var_I) if var_I and var_I > 0 else np.nan
    p_value = 2 * (1 - _norm_cdf(abs(z))) if not np.isnan(z) else np.nan
    return I, p_value


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def compute_all_metrics(y_true, y_pred, model_name, n_features=None,
                         y_pred_lower=None, y_pred_upper=None,
                         y_class_true=None, y_class_pred=None,
                         coords=None, train_time=None, sloocv_r2=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    r2 = r2_score(y_true, y_pred)
    adj_r2 = (1 - (1 - r2) * (n - 1) / (n - n_features - 1)
              if n_features and n - n_features - 1 > 0 else np.nan)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min()) * 100 if y_true.max() > y_true.min() else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pbias = 100 * np.sum(y_true - y_pred) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else np.nan
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if not np.isnan(r) else np.nan

    rmse_std = rmse / np.std(y_true) if np.std(y_true) > 0 else np.nan

    mean_obs = y_true.mean()
    ioa_denom = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
    ioa = 1 - ss_res / ioa_denom if ioa_denom > 0 else np.nan

    metrics = {
        'Model': model_name, 'R2': r2, 'Adj_R2': adj_r2, 'RMSE': rmse,
        'NRMSE_pct': nrmse, 'MAE': mae, 'MBE': mbe, 'NSE': nse, 'KGE': kge,
        'PBIAS_pct': pbias, 'IOA': ioa, 'RMSE_STD': rmse_std,
        'SLOOCV_R2': sloocv_r2, 'Training_Time_s': train_time,
    }

    if y_class_true is not None and y_class_pred is not None:
        metrics['OA'] = accuracy_score(y_class_true, y_class_pred)
        metrics['Kappa'] = cohen_kappa_score(y_class_true, y_class_pred)
        metrics['F1_macro'] = f1_score(y_class_true, y_class_pred, average='macro')

    if y_pred_lower is not None and y_pred_upper is not None:
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        mpiw = np.mean(y_pred_upper - y_pred_lower)
        metrics['Coverage_90pct'] = coverage
        metrics['MPIW'] = mpiw
        metrics['PICP'] = coverage

    if coords is not None:
        residuals = y_true - y_pred
        I, p = morans_i(residuals, coords)
        metrics['Moran_I_residuals'] = I
        metrics['Moran_I_pvalue'] = p

    return metrics


def run(results, y_test, features, class_test=None, coords_test=None, sloocv_results=None):
    """Build Table 4 -- main performance comparison, one row per model."""
    from utils import erosion_class
    rows = []
    sloocv_results = sloocv_results or {}
    class_map = {label: i + 1 for i, label in enumerate(config.EROSION_LABELS)}
    for name, r in results.items():
        y_pred = r['y_pred']
        y_class_pred = None
        if class_test is not None:
            # Regressors (esp. the DL models) can predict slightly negative
            # Soil_Loss for near-zero-erosion samples; EROSION_BINS starts
            # at 0, so an unclipped negative prediction falls in no bin and
            # pd.cut returns NaN -- clip to 0 before classifying, matching
            # how a real prediction map would be floored at zero.
            y_pred_nonneg = np.clip(y_pred, 0, None)
            y_class_pred = erosion_class(pd.Series(y_pred_nonneg)).map(class_map).astype(int).to_numpy()
        m = compute_all_metrics(
            y_test, y_pred, name, n_features=len(features),
            y_class_true=class_test, y_class_pred=y_class_pred,
            coords=coords_test, train_time=r.get('train_time'),
            sloocv_r2=sloocv_results.get(name, {}).get('R2'))
        rows.append(m)

    table = pd.DataFrame(rows).sort_values('R2', ascending=False).reset_index(drop=True)
    from utils import save_table
    save_table(table, "table04_model_performance_full.csv")
    print(table[['Model', 'R2', 'RMSE', 'NSE', 'KGE']].to_string(index=False))
    return table


if __name__ == "__main__":
    pass
