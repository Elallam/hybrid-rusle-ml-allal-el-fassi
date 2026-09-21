"""
utils.py -- Shared utility functions for Article J1 pipeline.

Houses the physically-consistent synthetic data generator (used only when
the real GIS-derived feature CSV is not yet available -- see
data/README_data.md for how to swap in real data), small IO/plotting
helpers reused across every numbered pipeline stage, and the FAO erosion
classifier.
"""
import os
import numpy as np
import pandas as pd

import config


def print_step(step, total, description):
    print(f"[STEP {step:>2}/{total}] {description}...")


def ensure_dirs():
    for d in (config.OUTPUT_DIR, config.FIGURES_DIR, config.TABLES_DIR,
              config.MODELS_DIR, config.RASTERS_DIR, config.QGIS_STYLES_DIR,
              config.RESULTS_DIR, config.PAPER_DIR, config.DATA_DIR):
        os.makedirs(d, exist_ok=True)


def confirm_saved(filepath):
    print(f"  Saved: {filepath}")


def erosion_class(soil_loss):
    """Map continuous Soil_Loss (t/ha/yr) to the 6 FAO/USLE susceptibility classes."""
    return pd.cut(soil_loss, bins=config.EROSION_BINS,
                   labels=config.EROSION_LABELS, include_lowest=True)


def set_plot_style():
    import matplotlib.pyplot as plt
    try:
        plt.style.use(config.STYLE)
    except (OSError, ValueError):
        plt.style.use('default')
    plt.rcParams.update({
        'font.size': config.FONT_SIZE,
        'axes.titlesize': config.FONT_SIZE + 1,
        'axes.labelsize': config.FONT_SIZE,
        'figure.dpi': 100,
        'savefig.dpi': config.DPI,
        'font.family': config.FONT_FAMILY,
    })


def save_fig(fig, filename_no_ext, formats=None):
    """Save a matplotlib figure in every format listed in config.FIG_FORMATS."""
    formats = formats or config.FIG_FORMATS
    paths = []
    for ext in formats:
        path = os.path.join(config.FIGURES_DIR, f"{filename_no_ext}.{ext}")
        fig.savefig(path, dpi=config.DPI, bbox_inches='tight')
        paths.append(path)
    confirm_saved(f"{filename_no_ext}.{{{','.join(formats)}}}")
    return paths


def save_table(df, filename, index=False):
    path = os.path.join(config.TABLES_DIR, filename)
    df.to_csv(path, index=index, encoding='utf-8')
    confirm_saved(path)
    return path


# ---------------------------------------------------------------------------
# Spatially-correlated Gaussian random field (Gaussian variogram)
# ---------------------------------------------------------------------------
def spatial_gaussian_field(coords, range_m, sill, rng):
    """Draw a zero-mean spatially-correlated Gaussian field at `coords`.

    Uses a Gaussian variogram covariance C(d) = sill * exp(-(d/range)^2) and
    a full Cholesky factorisation of the covariance matrix (feasible for the
    few-thousand sample sizes used here). This is what injects realistic
    spatial autocorrelation into the synthetic target, per CLAUDE.md 3.5.
    """
    n = len(coords)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    cov = sill * np.exp(-(d / range_m) ** 2) + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(n)
    return L @ z


# ---------------------------------------------------------------------------
# Shared physical feature model (RUSLE-consistent, 30-feature J1 schema)
# ---------------------------------------------------------------------------
def compute_physical_fields(nx, ny, rng):
    """Compute all 30 raw features + RUSLE A (noise-free) from normalized coords.

    Parameters
    ----------
    nx, ny : array-like in [0, 1]   Position within the watershed bbox.
    rng    : numpy.random.Generator

    Returns
    -------
    dict feature_name -> array (same shape as nx/ny), plus key 'A' for the
    noise-free RUSLE soil loss A = R x K x LS x C x P.
    """
    shape = np.shape(nx)

    # -- GROUP B: Topographic indices ---------------------------------------
    Elevation = np.clip(400 + 2100 * (0.55 * ny + 0.25 * nx + 0.20 * rng.random(shape)),
                         400, 2500)
    elev_norm = (Elevation - 400) / 2100.0
    Slope = np.clip(rng.gamma(2.0, 8.0, shape) + 12 * elev_norm, 0, 65)
    Aspect = rng.uniform(0, 360, shape)
    Aspect_sin = np.sin(np.radians(Aspect))
    Aspect_cos = np.cos(np.radians(Aspect))
    Curvature_Plan = rng.normal(0, 0.4, shape)
    Curvature_Profile = rng.normal(0, 0.4, shape)
    # Most hillslope pixels have low contributing area; only rare
    # drainage-line pixels have high accumulation -- a small exponential
    # scale (not 500) keeps that skew realistic instead of blowing up LS
    # (and therefore A_RUSLE) for the majority of the watershed.
    Flow_Accumulation = np.clip(rng.exponential(25, shape) + 1, 1, 400)
    slope_rad = np.radians(Slope)
    TWI = np.log(Flow_Accumulation / (np.tan(slope_rad) + 0.001))
    SPI = Flow_Accumulation * np.tan(slope_rad)
    TPI = rng.normal(0, 5, shape) + 3 * (elev_norm - 0.5)
    TRI = np.clip(Slope * rng.uniform(0.8, 1.3, shape), 0, None)
    Valley_Depth = np.clip(150 * (1 - elev_norm) + rng.normal(0, 15, shape), 0, None)
    # Capped at 25 (not the textbook-unbounded value): real RUSLE GIS
    # pipelines routinely cap LS this way since FlowAcc^0.4 otherwise spikes
    # to the hundreds along drainage cells (same issue documented for the
    # real watershed in the sibling C1/allal-erosion project).
    LS_Factor = np.clip(
        (Flow_Accumulation * config.RASTER_RES / 22.13) ** 0.4 *
        (np.sin(slope_rad) / 0.0896) ** 1.3,
        0, 25)

    # -- GROUP D: Climate (CHIRPS/ERA5-like, wetter & more seasonal north) --
    Annual_Rainfall = np.clip(450 + 750 * (0.6 * ny + 0.4 * rng.random(shape)), 450, 1200)
    Rainfall_Seasonality = np.clip(35 + 25 * rng.random(shape) + 0.01 * (900 - Annual_Rainfall),
                                    10, 90)
    Max_Monthly_Rain = np.clip(Annual_Rainfall * (0.18 + 0.10 * rng.random(shape)), 30, None)
    R_Factor = 0.0483 * Annual_Rainfall ** 1.61 / 17.0

    # -- GROUP C: Remote sensing indices (vegetation decreases with slope) --
    NDVI = np.clip(0.55 - 0.008 * Slope + 0.00035 * (Annual_Rainfall - 450)
                    - 0.15 * rng.random(shape), -0.1, 0.78)
    NDWI = np.clip(-0.2 - 0.3 * NDVI + rng.normal(0, 0.05, shape), -0.6, 0.3)
    NDBI = np.clip(0.05 - 0.1 * NDVI + rng.normal(0, 0.05, shape), -0.3, 0.4)
    EVI = np.clip(0.8 * NDVI + rng.normal(0, 0.03, shape), -0.1, 0.7)
    BSI = np.clip(0.3 - 0.5 * NDVI + rng.normal(0, 0.05, shape), -0.3, 0.6)
    SAVI = np.clip(1.5 * NDVI / (NDVI + 0.5), -0.2, 0.7)
    MSAVI = np.clip((2 * NDVI + 1 - np.sqrt(np.clip((2 * NDVI + 1) ** 2 - 8 * NDVI, 0, None))) / 2,
                     -0.2, 0.7)
    Albedo = np.clip(0.35 - 0.25 * NDVI + rng.normal(0, 0.02, shape), 0.05, 0.45)

    # -- GROUP E: Soil (SoilGrids-like) --------------------------------------
    Clay_Content = np.clip(rng.normal(25, 8, shape), 5, 55)
    Sand_Content = np.clip(rng.normal(40, 12, shape) - 0.3 * (Clay_Content - 25), 5, 80)
    SOC = np.clip(rng.normal(18, 6, shape) + 15 * np.clip(NDVI, 0, None), 2, 60)
    Bulk_Density = np.clip(1.5 - 0.004 * SOC + rng.normal(0, 0.05, shape), 1.0, 1.8)
    K_Factor = np.clip(0.15 + 0.004 * Clay_Content - 0.003 * SOC + rng.normal(0, 0.03, shape),
                        0.02, 0.65)

    # -- GROUP A: Land management / RUSLE --------------------------------------
    # Steeper NDVI decay than a naive exp(-2.5*NDVI): real RUSLE C-factor
    # lookup tables drop close to ~0.01-0.05 under dense cover, not ~0.15-0.3.
    C_Factor = np.clip(np.exp(-4.5 * NDVI), 0.005, 1.0)
    P_Factor = np.clip(rng.uniform(0.3, 1.0, shape) - 0.2 * (NDVI > 0.4), 0.1, 1.0)

    A_RUSLE = R_Factor * K_Factor * LS_Factor * C_Factor * P_Factor

    return {
        'R_Factor': R_Factor, 'K_Factor': K_Factor, 'LS_Factor': LS_Factor,
        'C_Factor': C_Factor, 'P_Factor': P_Factor, 'A_RUSLE': A_RUSLE,
        'Elevation': Elevation, 'Slope': Slope, 'Aspect': Aspect,
        'Aspect_sin': Aspect_sin, 'Aspect_cos': Aspect_cos,
        'Curvature_Plan': Curvature_Plan, 'Curvature_Profile': Curvature_Profile,
        'TWI': TWI, 'SPI': SPI, 'TPI': TPI, 'TRI': TRI,
        'Flow_Accumulation': Flow_Accumulation, 'Valley_Depth': Valley_Depth,
        'NDVI': NDVI, 'NDWI': NDWI, 'NDBI': NDBI, 'EVI': EVI, 'BSI': BSI,
        'SAVI': SAVI, 'MSAVI': MSAVI, 'Albedo': Albedo,
        'Annual_Rainfall': Annual_Rainfall,
        'Rainfall_Seasonality': Rainfall_Seasonality,
        'Max_Monthly_Rain': Max_Monthly_Rain,
        'Clay_Content': Clay_Content, 'Sand_Content': Sand_Content,
        'SOC': SOC, 'Bulk_Density': Bulk_Density,
        'A': A_RUSLE,
    }


def generate_synthetic_data(n_samples=None, noise_level=0.12, spatial_structure=True,
                             random_state=None):
    """Generate physically-realistic synthetic training data for J1.

    Mimics the Allal El Fassi sub-watershed so the hybrid pipeline can be
    developed/validated before the real GIS-extracted CSV lands (see
    data/README_data.md). Target Soil_Loss stays consistent with C1's RUSLE
    baseline (A = R x K x LS x C x P) plus multiplicative noise, with a
    Gaussian-variogram spatial random field added so nearby pixels are
    correlated -- this is what SLOOCV (Section 6) is designed to detect.
    """
    n_samples = n_samples or config.N_SAMPLES_SYN
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    rng = np.random.default_rng(random_state)

    bbox = config.WATERSHED_BBOX
    X_UTM = rng.uniform(bbox['xmin'], bbox['xmax'], n_samples)
    Y_UTM = rng.uniform(bbox['ymin'], bbox['ymax'], n_samples)
    nx = (X_UTM - bbox['xmin']) / (bbox['xmax'] - bbox['xmin'])
    ny = (Y_UTM - bbox['ymin']) / (bbox['ymax'] - bbox['ymin'])

    fields = compute_physical_fields(nx, ny, rng)
    A = fields.pop('A')

    noise = rng.normal(1.0, noise_level, n_samples)
    Soil_Loss = A * noise

    if spatial_structure:
        coords = np.column_stack([X_UTM, Y_UTM])
        spatial_noise = spatial_gaussian_field(coords, range_m=5000, sill=1.0, rng=rng)
        # Multiplicative spatial perturbation, scaled small relative to A's mean
        Soil_Loss = Soil_Loss * (1.0 + 0.15 * spatial_noise)

    Soil_Loss = np.clip(Soil_Loss, 0.01, None)

    df = pd.DataFrame({'X_UTM': X_UTM, 'Y_UTM': Y_UTM, **fields,
                        'Soil_Loss': Soil_Loss})
    df['Erosion_Class'] = erosion_class(df['Soil_Loss']).map(
        {label: i + 1 for i, label in enumerate(config.EROSION_LABELS)}).astype(int)

    # Inject a small fraction of missing values and target outliers so
    # preprocessing (Section 5, steps 3-4) has real, reproducible work to do.
    feature_cols = [c for c in config.RAW_FEATURES if c != 'Aspect'] + ['Aspect']
    feature_cols = [c for c in feature_cols if c in df.columns]
    missing_mask = rng.random(df[feature_cols].shape) < 0.01
    block = df[feature_cols].to_numpy()
    block[missing_mask] = np.nan
    df[feature_cols] = block

    n_outliers = max(1, int(0.005 * n_samples))
    outlier_idx = rng.choice(n_samples, n_outliers, replace=False)
    df.loc[outlier_idx, 'Soil_Loss'] *= rng.uniform(6, 10, n_outliers)
    df.loc[outlier_idx, 'Erosion_Class'] = erosion_class(df.loc[outlier_idx, 'Soil_Loss']).map(
        {label: i + 1 for i, label in enumerate(config.EROSION_LABELS)}).astype(int)

    return df
