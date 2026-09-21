"""
real_data.py -- Real-raster loading for Article J1.

Builds the full-resolution ~30-feature layer stack for the Allal El Fassi
watershed by combining:
  - C1 (allal-erosion): DEM, flow accumulation, LS/K/R/C/P factors, NDVI,
    12-band monthly CHIRPS, 8-band SoilGrids (sand/silt/clay/soc), A (target)
  - C2 (Comparative Evaluation of Machine Learning): 5-band Sentinel-2
    indices (NDWI, NDBI, EVI, BSI, SAVI)
  - This project's own GEE fetch (scripts/01_fetch_j1_extra_layers.py):
    MSAVI, Albedo, Bulk_Density (the only 3 layers not already available)

Everything else (Curvature_Plan/Profile, TPI, Valley_Depth, TWI, SPI, TRI,
Rainfall_Seasonality, Max_Monthly_Rain, Sand_Content, SOC) is DERIVED here
purely from those rasters -- no further external fetch needed.

A_RUSLE (= R*K*LS*C*P) is computed and returned for documentation (Table 1)
but is NOT a modeling feature -- see config.RUSLE_FEATURES's docstring:
Soil_Loss (the target) equals this same product by construction, so using
it as an input would make prediction a trivial identity mapping.
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

J1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C1_DIR = r"C:\Users\soufi\OneDrive\Bureau\phd\allal-erosion"
C2_DIR = r"C:\Users\soufi\OneDrive\Bureau\phd\Comparative Evaluation of Machine Learning"
BOUNDARY_PATH = os.path.join(C1_DIR, "data", "boundary", "allal_watershed.geojson")
CELL_SIZE = 30.0


def load_watershed_boundary(crs="EPSG:32629"):
    if not os.path.exists(BOUNDARY_PATH):
        return None
    import geopandas as gpd
    gdf = gpd.read_file(BOUNDARY_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:32629")
    return gdf.to_crs(crs)


def _read_band(path, band=1):
    with rasterio.open(path) as ds:
        return ds.read(band).astype(np.float64), ds.profile


def _reproject_to_master(src_path, master_profile, band=1, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        dst = np.full((master_profile['height'], master_profile['width']), np.nan, dtype=np.float64)
        reproject(source=rasterio.band(src, band), destination=dst,
                   src_transform=src.transform, src_crs=src.crs,
                   dst_transform=master_profile['transform'], dst_crs=master_profile['crs'],
                   resampling=resampling, dst_nodata=np.nan)
        return dst


def _compute_terrain_factors(dem, cell_size=CELL_SIZE):
    """Slope, Aspect, plan & profile curvature via finite differences
    (Zevenbergen & Thorne, 1987 -- standard 2nd-derivative formulation)."""
    dzdrow, dzdcol = np.gradient(dem, cell_size)
    gy, gx = -dzdrow, dzdcol  # row increases southward in a north-up raster

    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    slope_deg = np.degrees(slope_rad)

    aspect_deg = 90.0 - np.degrees(np.arctan2(gy, -gx))
    aspect_deg = np.where((gx == 0) & (gy == 0), 0.0, np.mod(aspect_deg + 360, 360))

    d2zdrow2 = np.gradient(dzdrow, cell_size, axis=0)
    d2zdcol2 = np.gradient(dzdcol, cell_size, axis=1)
    d2zdrowcol = np.gradient(dzdrow, cell_size, axis=1)

    p = gx ** 2 + gy ** 2
    q = p + 1
    # Profile curvature: rate of change of slope in the direction of steepest descent
    profile = np.where(p > 1e-9,
                        -(d2zdcol2 * gx ** 2 + 2 * d2zdrowcol * gx * gy + d2zdrow2 * gy ** 2) /
                        (p * np.power(q, 1.5) + 1e-12),
                        0.0) * 100
    # Plan curvature: rate of change of aspect along a contour
    plan = np.where(p > 1e-9,
                     -(d2zdcol2 * gy ** 2 - 2 * d2zdrowcol * gx * gy + d2zdrow2 * gx ** 2) /
                     (np.power(p, 1.5) + 1e-12),
                     0.0) * 100

    return slope_deg, aspect_deg, plan, profile


def _compute_tri(dem):
    padded = np.pad(dem, 1, mode='edge')
    total = np.zeros_like(dem)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr:1 + dr + dem.shape[0], 1 + dc:1 + dc + dem.shape[1]]
            total += (dem - neighbor) ** 2
    return np.sqrt(total)


def _compute_twi_spi(flow_acc, slope_deg, cell_size=CELL_SIZE):
    slope_rad = np.radians(np.clip(slope_deg, 0.1, None))
    sca = np.clip(flow_acc, 1, None) * cell_size
    twi = np.log(sca / np.tan(slope_rad))
    spi = np.log(sca * np.tan(slope_rad) + 1e-3)
    return twi, spi


def _uniform_filter_nan_safe(arr, size):
    """Mean filter that ignores NaNs at the watershed edge (avoids bleeding
    NaN into valid pixels near the mask boundary)."""
    from scipy.ndimage import uniform_filter
    filled = np.nan_to_num(arr, nan=0.0)
    valid = (~np.isnan(arr)).astype(np.float64)
    num = uniform_filter(filled, size=size, mode='nearest')
    den = uniform_filter(valid, size=size, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    return np.where(den > 0, out, np.nan)


def _compute_tpi_valley_depth(dem, small_window=3, large_window=21):
    """TPI: elevation minus local mean (small window). Valley_Depth: local
    relief proxy = large-window neighbourhood mean minus elevation (positive
    in valleys, matches config's "Local relief" description)."""
    local_mean = _uniform_filter_nan_safe(dem, small_window)
    tpi = dem - local_mean
    regional_mean = _uniform_filter_nan_safe(dem, large_window)
    valley_depth = np.clip(regional_mean - dem, 0, None)
    return tpi, valley_depth


def real_data_available(year=2025):
    required = [
        os.path.join(C1_DIR, "data/interim/dem.tif"),
        os.path.join(C1_DIR, f"data/processed/A_{year}.tif"),
        os.path.join(C2_DIR, "data", "raw", f"sentinel2_indices_{year}.tif"),
        os.path.join(J1_DIR, "data", "raw", f"j1_extra_layers_{year}.tif"),
    ]
    return all(os.path.exists(p) for p in required)


def build_watershed_layers(year=2025, ls_cap_percentile=99):
    """Build the full-resolution real feature layer stack + valid-pixel mask.

    Returns
    -------
    layers : dict[str, np.ndarray]   One 2D array per RAW_FEATURES + TARGET.
    mask : np.ndarray[bool]
    master_profile : dict            rasterio profile of the master grid (DEM)
    """
    dem, master_profile = _read_band(os.path.join(C1_DIR, "data/interim/dem.tif"))
    slope, aspect, curv_plan, curv_profile = _compute_terrain_factors(dem)
    tri = _compute_tri(dem)
    tpi, valley_depth = _compute_tpi_valley_depth(dem)

    flow_acc = _reproject_to_master(os.path.join(C1_DIR, "data/interim/facc.tif"), master_profile)
    twi, spi = _compute_twi_spi(flow_acc, slope)

    ls_factor, _ = _read_band(os.path.join(C1_DIR, "data/processed/LS.tif"))
    k_factor, _ = _read_band(os.path.join(C1_DIR, "data/processed/K.tif"))
    r_factor, _ = _read_band(os.path.join(C1_DIR, f"data/processed/R_{year}.tif"))
    c_factor, _ = _read_band(os.path.join(C1_DIR, f"data/processed/C_{year}.tif"))
    p_factor, _ = _read_band(os.path.join(C1_DIR, f"data/processed/P_{year}.tif"))
    soil_loss, _ = _read_band(os.path.join(C1_DIR, f"data/processed/A_{year}.tif"))
    ndvi, _ = _read_band(os.path.join(C1_DIR, f"data/interim/ndvi_{year}.tif"))

    valid_ls = ls_factor[~np.isnan(ls_factor)]
    ls_cap = np.percentile(valid_ls, ls_cap_percentile)
    ls_factor = np.clip(ls_factor, 0, ls_cap)

    # -- Climate: 12-band monthly CHIRPS -> annual total, seasonality, max month --
    with rasterio.open(os.path.join(C1_DIR, f"data/interim/chirps_{year}.tif")) as ds:
        monthly = ds.read().astype(np.float64)  # (12, H, W)
    annual_rainfall = monthly.sum(axis=0)
    monthly_mean = monthly.mean(axis=0)
    monthly_std = monthly.std(axis=0)
    rainfall_seasonality = np.where(monthly_mean > 0, 100 * monthly_std / monthly_mean, np.nan)
    max_monthly_rain = monthly.max(axis=0)

    # -- Soil: 8-band SoilGrids (sand/silt/clay/soc at 0-5cm, 5-15cm) --
    with rasterio.open(os.path.join(C1_DIR, "data/interim/soilgrids.tif")) as ds:
        sand_0_5, sand_5_15 = ds.read(1).astype(np.float64), ds.read(2).astype(np.float64)
        clay_0_5, clay_5_15 = ds.read(5).astype(np.float64), ds.read(6).astype(np.float64)
        soc_0_5, soc_5_15 = ds.read(7).astype(np.float64), ds.read(8).astype(np.float64)
    clay_content = (clay_0_5 + clay_5_15) / 2.0 / 10.0
    sand_content = (sand_0_5 + sand_5_15) / 2.0 / 10.0
    soc = (soc_0_5 + soc_5_15) / 2.0 / 10.0

    # -- Sentinel-2 indices: C2's 5 bands + this project's MSAVI/Albedo --
    s2_path = os.path.join(C2_DIR, "data", "raw", f"sentinel2_indices_{year}.tif")
    ndwi = _reproject_to_master(s2_path, master_profile, band=1)
    ndbi = _reproject_to_master(s2_path, master_profile, band=2)
    evi = _reproject_to_master(s2_path, master_profile, band=3)
    bsi = _reproject_to_master(s2_path, master_profile, band=4)
    savi = _reproject_to_master(s2_path, master_profile, band=5)

    extra_path = os.path.join(J1_DIR, "data", "raw", f"j1_extra_layers_{year}.tif")
    msavi = _reproject_to_master(extra_path, master_profile, band=1)
    albedo = _reproject_to_master(extra_path, master_profile, band=2)
    bd_0_5 = _reproject_to_master(extra_path, master_profile, band=3)
    bd_5_15 = _reproject_to_master(extra_path, master_profile, band=4)
    bulk_density = (bd_0_5 + bd_5_15) / 2.0

    a_rusle = r_factor * k_factor * ls_factor * c_factor * p_factor

    layers = {
        'R_Factor': r_factor, 'K_Factor': k_factor, 'LS_Factor': ls_factor,
        'C_Factor': c_factor, 'P_Factor': p_factor, 'A_RUSLE': a_rusle,
        'Elevation': dem, 'Slope': slope, 'Aspect': aspect,
        'Curvature_Plan': curv_plan, 'Curvature_Profile': curv_profile,
        'TWI': twi, 'SPI': spi, 'TPI': tpi, 'TRI': tri,
        'Flow_Accumulation': flow_acc, 'Valley_Depth': valley_depth,
        'NDVI': ndvi, 'NDWI': ndwi, 'NDBI': ndbi, 'EVI': evi, 'BSI': bsi,
        'SAVI': savi, 'MSAVI': msavi, 'Albedo': albedo,
        'Annual_Rainfall': annual_rainfall,
        'Rainfall_Seasonality': rainfall_seasonality,
        'Max_Monthly_Rain': max_monthly_rain,
        'Clay_Content': clay_content, 'Sand_Content': sand_content,
        'SOC': soc, 'Bulk_Density': bulk_density,
        'Soil_Loss': soil_loss,
    }

    mask = np.ones(dem.shape, dtype=bool)
    for arr in layers.values():
        mask &= np.isfinite(arr)

    return layers, mask, master_profile, ls_cap


def build_training_csv(output_path, year=2025, n_samples=None, random_state=None):
    """Stratified-sample pixels from the real watershed layer stack into a
    training CSV (config's RAW_FEATURES + TARGET + TARGET_CLASS + coords).
    Shared by scripts/02_build_real_features_csv.py and 01_data_loading.py
    (which calls this automatically if real data is available but the CSV
    hasn't been built yet)."""
    import config
    n_samples = n_samples or config.N_SAMPLES_SYN
    random_state = random_state if random_state is not None else config.RANDOM_STATE

    layers, mask, master_profile, ls_cap = build_watershed_layers(year=year)
    print(f"  Valid watershed pixels available: {int(mask.sum())}; "
          f"LS_Factor capped at 99th pct = {ls_cap:.2f}")

    rng = np.random.default_rng(random_state)
    rows, cols = np.where(mask)
    soil_loss_valid = layers[config.TARGET][mask]
    strat_class = pd.cut(soil_loss_valid, bins=config.EROSION_BINS,
                          labels=config.EROSION_LABELS, include_lowest=True)

    idx_all = np.arange(len(rows))
    sample_idx = []
    per_class_target = max(1, n_samples // len(config.EROSION_LABELS))
    for label in config.EROSION_LABELS:
        class_idx = idx_all[strat_class == label]
        n_take = min(per_class_target, len(class_idx))
        if n_take > 0:
            sample_idx.extend(rng.choice(class_idx, n_take, replace=False))
    remaining = n_samples - len(sample_idx)
    if remaining > 0:
        leftover = np.setdiff1d(idx_all, sample_idx)
        sample_idx.extend(rng.choice(leftover, min(remaining, len(leftover)), replace=False))
    sample_idx = np.array(sample_idx)

    sel_rows, sel_cols = rows[sample_idx], cols[sample_idx]
    xs, ys = rasterio.transform.xy(master_profile['transform'], sel_rows, sel_cols)

    data = {'X_UTM': xs, 'Y_UTM': ys}
    for name in config.RAW_FEATURES:
        data[name] = layers[name][sel_rows, sel_cols]
    data[config.TARGET] = layers[config.TARGET][sel_rows, sel_cols]
    df = pd.DataFrame(data)

    class_map = {label: i + 1 for i, label in enumerate(config.EROSION_LABELS)}
    df[config.TARGET_CLASS] = pd.cut(df[config.TARGET], bins=config.EROSION_BINS,
                                      labels=config.EROSION_LABELS,
                                      include_lowest=True).map(class_map).astype(int)

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved: {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    return df


def build_downsampled_grid(year=2025, target_dim=None):
    """Downsample the full-resolution watershed layer stack for tractable
    raster prediction / SHAP interpolation (5M+ pixels at native 30m is too
    many to predict+SHAP-interpolate repeatedly). Same stride-subsampling
    approach as C2's 09_spatial_prediction.build_prediction_grid_real.

    Returns
    -------
    bands : dict[str, np.ndarray]   2D arrays, config.RAW_FEATURES order
    mask : np.ndarray[bool]         valid-pixel mask, same shape
    transform : rasterio.Affine     transform for the DOWNSAMPLED grid
    """
    import config
    from rasterio.transform import Affine
    target_dim = target_dim or config.RASTER_DEMO_DIM
    layers, mask, master_profile, _ = build_watershed_layers(year=year)
    ny_full, nx_full = mask.shape
    stride = max(1, round(max(ny_full, nx_full) / target_dim))

    bands_ds = {k: v[::stride, ::stride] for k, v in layers.items() if k in config.RAW_FEATURES}
    mask_ds = mask[::stride, ::stride]

    src_transform = master_profile['transform']
    ds_transform = src_transform * Affine.scale(stride, stride)

    for name in bands_ds:
        bands_ds[name] = np.where(mask_ds, bands_ds[name], np.nan)

    print(f"  Real grid downsampled: {mask_ds.shape} (stride={stride}), "
          f"{int(mask_ds.sum())} valid pixels, cell={CELL_SIZE * stride:.0f}m")
    return bands_ds, mask_ds, ds_transform
