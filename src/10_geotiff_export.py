"""
10_geotiff_export.py -- Raster prediction + GeoTIFF export module.

CLAUDE.md Section 11. If data/feature_stack.tif doesn't exist yet, it's
built automatically -- from the real watershed raster stack
(src/real_data.py) when real GIS data is available (see
data/README_data.md), else a physically consistent synthetic stack over
WATERSHED_BBOX (Section 3.5) so the prediction -> GeoTIFF -> QGIS-style
pipeline is always runnable end to end regardless of which machine it's on.
"""
import os
import importlib.util
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

import config
from utils import compute_physical_fields, confirm_saved, erosion_class

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SRC_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def array_to_geotiff(array, transform, crs, output_path, nodata=None, band_names=None):
    """Write a 2D (or 3D band-first) numpy array to GeoTIFF using the given
    affine transform/CRS. NaNs are written as `nodata`. `band_names`, if
    given, is stored as each band's description (so a later read can
    recover feature order without relying on a fixed convention)."""
    nodata = config.NODATA_VALUE if nodata is None else nodata
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    count, height, width = arr.shape
    arr = np.where(np.isnan(arr), nodata, arr)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width,
                        count=count, dtype='float32', crs=crs, transform=transform,
                        nodata=nodata, compress='lzw') as dst:
        for b in range(count):
            dst.write(arr[b], b + 1)
            if band_names and b < len(band_names):
                dst.set_band_description(b + 1, band_names[b])
    confirm_saved(output_path)
    return output_path


def build_synthetic_raster_stack(output_path=None):
    """Generate a physically-consistent multi-band raster (RAW_FEATURES
    order) over WATERSHED_BBOX, masked to an elliptical pseudo-watershed
    boundary so downstream maps look like a real sub-watershed."""
    output_path = output_path or config.RASTER_STACK
    bbox = config.WATERSHED_BBOX
    width_m = bbox['xmax'] - bbox['xmin']
    height_m = bbox['ymax'] - bbox['ymin']
    long_edge_px = config.RASTER_DEMO_DIM
    res = max(width_m, height_m) / long_edge_px
    nx = max(2, int(round(width_m / res)))
    ny = max(2, int(round(height_m / res)))

    xs = np.linspace(bbox['xmin'] + res / 2, bbox['xmax'] - res / 2, nx)
    ys = np.linspace(bbox['ymax'] - res / 2, bbox['ymin'] + res / 2, ny)  # north -> south
    XX, YY = np.meshgrid(xs, ys)
    nx_norm = (XX - bbox['xmin']) / width_m
    ny_norm = (YY - bbox['ymin']) / height_m

    rng = np.random.default_rng(config.RANDOM_STATE)
    fields = compute_physical_fields(nx_norm, ny_norm, rng)
    fields.pop('A')

    # Elliptical pseudo-watershed mask (visual stand-in for the real boundary)
    mask = (((nx_norm - 0.5) ** 2) / (0.48 ** 2) + ((ny_norm - 0.5) ** 2) / (0.42 ** 2)) <= 1.0

    band_order = [c for c in config.RAW_FEATURES if c in fields]
    bands = []
    for name in band_order:
        arr = fields[name].astype(np.float32)
        arr = np.where(mask, arr, np.nan)
        bands.append(arr)
    bands = np.stack(bands, axis=0)

    transform = from_bounds(bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'], nx, ny)
    array_to_geotiff(bands, transform, config.CRS, output_path, band_names=band_order)

    meta = {'band_order': band_order, 'transform': transform, 'shape': (ny, nx), 'mask': mask}
    return output_path, meta


def build_real_raster_stack(output_path=None, year=2025):
    """Build feature_stack.tif from the real watershed raster stack
    (real_data.build_downsampled_grid), masked to the true watershed extent
    -- used automatically instead of the synthetic demo stack whenever real
    GIS data is available (see data/README_data.md)."""
    output_path = output_path or config.RASTER_STACK
    real_data = _load_sibling("real_data_geo10", "real_data.py")
    bands_dict, mask, transform = real_data.build_downsampled_grid(year=year)

    band_order = [c for c in config.RAW_FEATURES if c in bands_dict]
    bands = np.stack([bands_dict[name].astype(np.float32) for name in band_order], axis=0)

    array_to_geotiff(bands, transform, config.CRS, output_path, band_names=band_order)
    meta = {'band_order': band_order, 'transform': transform, 'shape': mask.shape, 'mask': mask}
    return output_path, meta


def load_raster_stack(path):
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
        arr = np.where(arr == src.nodata, np.nan, arr) if src.nodata is not None else arr
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
        descriptions = src.descriptions
    return arr, transform, crs, shape, descriptions


def _prepare_pixel_dataframe(bands_dict):
    """bands_dict: {feature_name: 2D array}. Applies the SAME aspect
    circular transform + engineered features used in training (02/03)."""
    pp = _load_sibling("pp10", "02_preprocessing.py")
    fe = _load_sibling("fe10", "03_feature_engineering.py")

    flat = {k: v.ravel() for k, v in bands_dict.items()}
    df = pd.DataFrame(flat)
    df = pp.aspect_circular_transform(df)
    df = fe.engineer_features(df)
    return df


def prepare_raster_matrix(feature_names, scaler, raster_stack_path=None):
    """Load (or synthesize) the raster stack, apply the same preprocessing/
    feature-engineering used at training time, and return the scaled
    per-pixel feature matrix ready for model.predict, plus enough metadata
    (valid_mask, transform, shape) to scatter predictions back onto the grid."""
    raster_stack_path = raster_stack_path or config.RASTER_STACK

    if not os.path.exists(raster_stack_path):
        real_data = _load_sibling("real_data_prep10", "real_data.py")
        if real_data.real_data_available():
            print(f"  {raster_stack_path} not found -- building it from the real watershed raster stack.")
            raster_stack_path, meta = build_real_raster_stack(raster_stack_path)
        else:
            print(f"  {raster_stack_path} not found -- generating synthetic demo raster stack.")
            raster_stack_path, meta = build_synthetic_raster_stack(raster_stack_path)
        band_order, transform, shape = meta['band_order'], meta['transform'], meta['shape']
        with rasterio.open(raster_stack_path) as src:
            arr = src.read()
            arr = np.where(arr == src.nodata, np.nan, arr)
    else:
        arr, transform, crs, shape, descriptions = load_raster_stack(raster_stack_path)
        band_order = list(descriptions) if descriptions and descriptions[0] else config.RAW_FEATURES[:arr.shape[0]]

    bands_dict = {name: arr[i] for i, name in enumerate(band_order)}
    df = _prepare_pixel_dataframe(bands_dict)

    valid_mask = df[feature_names].notna().all(axis=1).to_numpy()
    X_valid = df.loc[valid_mask, feature_names].to_numpy(dtype=float)
    X_scaled = scaler.transform(X_valid)

    return {'X_scaled': X_scaled, 'valid_mask': valid_mask, 'transform': transform,
            'shape': shape, 'n_pixels': len(df)}


def predict_raster(model, feature_names, scaler, output_path, raster_stack_path=None,
                    batch_size=None, predict_fn=None, prepared=None):
    """Predict Soil_Loss over the full raster grid and save as GeoTIFF.

    Parameters
    ----------
    model         : trained model exposing .predict(X) (or pass predict_fn)
    feature_names : final selected feature list, in training column order
    scaler        : fitted scaler (StandardScaler for ML models)
    predict_fn    : optional callable(X_scaled) -> y_pred, overrides model.predict
                    (used for DL / stacking models needing a specific call signature)
    prepared      : optional pre-computed prepare_raster_matrix(...) output, to
                     avoid reloading/reprocessing the raster stack repeatedly
    """
    batch_size = batch_size or config.BATCH_SIZE_RASTER
    prep = prepared or prepare_raster_matrix(feature_names, scaler, raster_stack_path)
    X_scaled, valid_mask = prep['X_scaled'], prep['valid_mask']
    transform, shape, n_pixels = prep['transform'], prep['shape'], prep['n_pixels']

    predict = predict_fn or model.predict
    n = len(X_scaled)
    preds = np.full(n, np.nan)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = predict(X_scaled[start:end])
        preds[start:end] = np.asarray(out).flatten()

    full_pred = np.full(n_pixels, np.nan)
    full_pred[valid_mask] = preds
    pred_raster = full_pred.reshape(shape)

    array_to_geotiff(pred_raster, transform, config.CRS, output_path)
    return pred_raster, transform, shape


def classify_erosion_raster(soil_loss_array, transform, output_path, qml_path=None):
    """Apply FAO classification thresholds -> integer raster (1-6) + .qml style."""
    classes = np.full(soil_loss_array.shape, np.nan)
    valid = ~np.isnan(soil_loss_array)
    # Clip to 0 first: EROSION_BINS starts at 0, and a model can predict
    # slightly negative Soil_Loss for near-zero-erosion pixels, which would
    # otherwise fall in no bin and come back as NaN.
    soil_loss_nonneg = np.clip(soil_loss_array[valid], 0, None)
    labels = erosion_class(pd.Series(soil_loss_nonneg)).map(
        {lab: i + 1 for i, lab in enumerate(config.EROSION_LABELS)})
    classes[valid] = labels.to_numpy()

    array_to_geotiff(classes, transform, config.CRS, output_path)

    qml_path = qml_path or os.path.join(config.QGIS_STYLES_DIR, "erosion_class_style.qml")
    _write_qml_style(qml_path)
    return classes


def _write_qml_style(qml_path):
    entries = "\n".join(
        f'        <paletteEntry value="{i+1}" color="{color}" label="{label}" alpha="255"/>'
        for i, (label, color) in enumerate(zip(config.EROSION_LABELS, config.EROSION_COLORS)))
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    os.makedirs(os.path.dirname(qml_path), exist_ok=True)
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml)
    confirm_saved(qml_path)


def export_uncertainty_rasters(mean_flat, lower_flat, upper_flat, valid_mask, shape,
                                transform, output_dir=None):
    output_dir = output_dir or config.RASTERS_DIR

    def _expand(flat_valid):
        full = np.full(shape[0] * shape[1], np.nan)
        full[valid_mask] = flat_valid
        return full.reshape(shape)

    mean_r = _expand(mean_flat)
    lower_r = _expand(lower_flat)
    upper_r = _expand(upper_flat)
    width_r = upper_r - lower_r
    cv_r = np.where(mean_r > 0, 100 * width_r / (2 * mean_r), np.nan)

    array_to_geotiff(mean_r, transform, config.CRS, os.path.join(output_dir, "uncertainty_mean.tif"))
    array_to_geotiff(lower_r, transform, config.CRS, os.path.join(output_dir, "uncertainty_lower90.tif"))
    array_to_geotiff(upper_r, transform, config.CRS, os.path.join(output_dir, "uncertainty_upper90.tif"))
    array_to_geotiff(width_r, transform, config.CRS, os.path.join(output_dir, "uncertainty_width.tif"))
    array_to_geotiff(cv_r, transform, config.CRS, os.path.join(output_dir, "uncertainty_cv.tif"))
    return {'mean': mean_r, 'lower': lower_r, 'upper': upper_r, 'width': width_r, 'cv': cv_r}


def raster_statistics(array, pixel_area_km2=None, transform=None, is_class=False):
    valid = array[~np.isnan(array)]
    stats = {
        'min': float(np.min(valid)) if valid.size else np.nan,
        'max': float(np.max(valid)) if valid.size else np.nan,
        'mean': float(np.mean(valid)) if valid.size else np.nan,
        'std': float(np.std(valid)) if valid.size else np.nan,
        'p05': float(np.percentile(valid, 5)) if valid.size else np.nan,
        'p50': float(np.percentile(valid, 50)) if valid.size else np.nan,
        'p95': float(np.percentile(valid, 95)) if valid.size else np.nan,
    }
    if is_class and transform is not None:
        px_area_km2 = abs(transform.a * transform.e) / 1e6
        rows = []
        for i, label in enumerate(config.EROSION_LABELS, 1):
            n_px = int(np.sum(array == i))
            rows.append({'Class': label, 'n_pixels': n_px, 'Area_km2': n_px * px_area_km2,
                         'Pct': 100 * n_px / valid.size if valid.size else np.nan})
        return stats, pd.DataFrame(rows)
    return stats, None


if __name__ == "__main__":
    pass
