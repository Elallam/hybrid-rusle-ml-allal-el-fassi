"""
11_shap_analysis.py -- Standard + spatial SHAP analysis (J1 novelty).

CLAUDE.md Section 8. Standard SHAP (XGBoost vs Stacking Ensemble beeswarm +
dependence plots), spatial SHAP maps (IDW-interpolated GeoTIFFs per top-5
feature -- shows WHERE each factor drives erosion), SHAP interaction values
(top feature pairs), and force-plot data for 3 representative points.
"""
import os
import importlib.util
import numpy as np
import shap
from scipy.spatial import cKDTree
from rasterio.transform import from_bounds

import config
from utils import confirm_saved

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SRC_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_shap_values(model, X, background=None, model_type='tree'):
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value
    else:
        bg = background if background is not None else shap.sample(X, min(50, len(X)))
        explainer = shap.Explainer(model.predict, bg)
        exp = explainer(X)
        shap_values = exp.values
        base_value = np.mean(exp.base_values) if hasattr(exp, 'base_values') else np.nan
    base_value = float(np.ravel(base_value)[0]) if np.ndim(base_value) else float(base_value)
    return shap_values, base_value


def shap_interaction_values(xgb_model, X_test, feature_names, top_pairs=5):
    explainer = shap.TreeExplainer(xgb_model)
    inter = explainer.shap_interaction_values(X_test)
    mean_abs_inter = np.abs(inter).mean(axis=0)
    np.fill_diagonal(mean_abs_inter, 0)
    n = len(feature_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((feature_names[i], feature_names[j], mean_abs_inter[i, j]))
    pairs.sort(key=lambda t: -t[2])
    return pairs[:top_pairs], inter


def idw_interpolate(points_xy, values, grid_x, grid_y, power=2, k=12):
    k = min(k, len(points_xy))
    tree = cKDTree(points_xy)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    dist, idx = tree.query(grid_pts, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    dist = np.where(dist == 0, 1e-6, dist)
    weights = 1.0 / dist ** power
    weights /= weights.sum(axis=1, keepdims=True)
    interp = (weights * values[idx]).sum(axis=1)
    return interp.reshape(grid_x.shape)


def _interpolation_grid(coords_test, pad_frac=0.05):
    """Grid bounds derived from the ACTUAL test-point coordinates (not a
    hardcoded WATERSHED_BBOX) -- correct for both real data (a completely
    different extent than the synthetic bbox) and synthetic data alike."""
    xmin, ymin = coords_test.min(axis=0)
    xmax, ymax = coords_test.max(axis=0)
    pad_x, pad_y = (xmax - xmin) * pad_frac, (ymax - ymin) * pad_frac
    xmin, xmax, ymin, ymax = xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y

    width_m, height_m = xmax - xmin, ymax - ymin
    dim = config.RASTER_DEMO_DIM
    res = max(width_m, height_m) / dim
    nx = max(2, int(round(width_m / res)))
    ny = max(2, int(round(height_m / res)))
    xs = np.linspace(xmin + res / 2, xmax - res / 2, nx)
    ys = np.linspace(ymax - res / 2, ymin + res / 2, ny)
    XX, YY = np.meshgrid(xs, ys)
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    return XX, YY, transform


def compute_spatial_shap(shap_values, coords_test, feature_names, output_dir=None, top_n=5):
    output_dir = output_dir or config.RASTERS_DIR
    geotiff = _load_sibling("geo11", "10_geotiff_export.py")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_idx]

    XX, YY, transform = _interpolation_grid(coords_test)
    shap_rasters = {}
    for i, feat in zip(top_idx, top_features):
        grid = idw_interpolate(coords_test, shap_values[:, i], XX, YY)
        path = os.path.join(output_dir, f"shap_map_{feat}.tif")
        geotiff.array_to_geotiff(grid, transform, config.CRS, path)
        shap_rasters[feat] = {'path': path, 'grid': grid}

    return top_features, shap_rasters


def representative_points(y_pred, n=3):
    """Indices of a high-, moderate- and low-erosion test point for force plots."""
    order = np.argsort(y_pred)
    return {
        'low': int(order[len(order) // 10]),
        'moderate': int(order[len(order) // 2]),
        'high': int(order[-max(1, len(order) // 10)]),
    }


def run(xgb_model, stacking_model, X_test, feature_names, coords_test, y_pred_for_repr=None,
        max_kernel_samples=150):
    print("  Computing SHAP values for XGBoost (TreeExplainer)...")
    shap_xgb, base_xgb = compute_shap_values(xgb_model, X_test, model_type='tree')

    print(f"  Computing SHAP values for Stacking Ensemble (model-agnostic Explainer, "
          f"subsampled to {min(max_kernel_samples, len(X_test))} points -- KernelExplainer "
          f"is O(n_samples x n_features), too slow on the full test set)...")
    rng = np.random.default_rng(config.RANDOM_STATE)
    kernel_idx = rng.choice(len(X_test), min(max_kernel_samples, len(X_test)), replace=False)
    shap_stack, base_stack = compute_shap_values(stacking_model, X_test[kernel_idx], model_type='kernel')

    print("  Computing SHAP interaction values (XGBoost, top feature pairs)...")
    top_pairs, interaction_values = shap_interaction_values(xgb_model, X_test, feature_names)
    for f1, f2, val in top_pairs:
        print(f"    {f1} x {f2}: mean|interaction|={val:.4f}")

    print("  Building spatial SHAP maps (top-5 features, IDW-interpolated)...")
    top_features, shap_rasters = compute_spatial_shap(shap_xgb, coords_test, feature_names)

    repr_idx = representative_points(y_pred_for_repr if y_pred_for_repr is not None
                                      else shap_xgb.sum(axis=1) + base_xgb)

    return {
        'shap_xgb': shap_xgb, 'base_xgb': base_xgb,
        'shap_stack': shap_stack, 'base_stack': base_stack, 'X_test_kernel': X_test[kernel_idx],
        'top_pairs': top_pairs, 'interaction_values': interaction_values,
        'top_features': top_features, 'shap_rasters': shap_rasters,
        'representative_points': repr_idx,
    }


if __name__ == "__main__":
    pass
