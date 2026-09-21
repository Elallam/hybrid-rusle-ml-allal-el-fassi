"""
config.py — All parameters for Article J1
"An Integrated Hybrid Framework Combining RUSLE, Remote Sensing, and
Ensemble Machine Learning for Soil Erosion Susceptibility Mapping in a
Semi-Arid Mediterranean Sub-Watershed"

Builds on C1 (RUSLE baseline) + C2 (ML comparison). See CLAUDE_Article_J1.md.
"""
import os

# ── Identity ───────────────────────────────────────────────────
ARTICLE_ID = "J1"
STUDY_AREA = "Allal El Fassi Sub-Watershed, Sebou Basin, Morocco"
TARGET_JOURNAL = "Catena"
CRS = "EPSG:32629"

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV = os.path.join(DATA_DIR, "features_j1_allalelfassi.csv")
RASTER_STACK = os.path.join(DATA_DIR, "feature_stack.tif")
WATERSHED_SHP = os.path.join(DATA_DIR, "watershed_boundary.shp")
DEM_PATH = os.path.join(DATA_DIR, "dem_watershed.tif")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RASTERS_DIR = os.path.join(OUTPUT_DIR, "rasters")
QGIS_STYLES_DIR = os.path.join(OUTPUT_DIR, "qgis_styles")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PAPER_DIR = os.path.join(BASE_DIR, "paper")

for _d in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR,
           RASTERS_DIR, QGIS_STYLES_DIR, RESULTS_DIR, PAPER_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Random and Reproducibility ─────────────────────────────────
RANDOM_STATE = 42
TF_SEED = 42

# ── Data ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
N_FOLDS_BLOCK = 5           # spatial block CV folds
BUFFER_DIST = 3000          # SLOOCV buffer in meters (3 km)
N_BOOTSTRAP = 100           # bootstrap iterations for uncertainty
N_SAMPLES_SYN = 3000        # synthetic data size
SLOOCV_MAX_POINTS = 300     # SLOOCV is O(n^2)-ish; subsample test points if n exceeds this

# ── Watershed bbox (UTM 29N, meters) — used ONLY by the synthetic-data
# fallback (generate_synthetic_data, build_synthetic_raster_stack) when real
# GIS data isn't available; matches the REAL Allal El Fassi watershed's
# actual extent (confirmed from data/features_j1_allalelfassi.csv once real
# data was wired in 2026-09-16) so a synthetic run on another machine still
# produces a plausible bbox. Real runs never read this -- real coordinates
# come from the actual rasters via src/real_data.py. ──
WATERSHED_BBOX = {
    "xmin": 854000, "xmax": 947000,
    "ymin": 3680000, "ymax": 3765000,
}

# ── Feature Groups ─────────────────────────────────────────────
# A_RUSLE is deliberately EXCLUDED from the modeling features: it equals
# R*K*LS*C*P by construction, and Soil_Loss (the target) IS that same
# product (Section 3.2: "SOURCE: RUSLE A from C1 paper") -- for real data
# specifically, A_RUSLE and Soil_Loss are numerically identical, so
# including it as an input feature would make prediction a trivial
# identity mapping (R2~1.0) rather than genuine learning. It's still
# computed/reported (RAW_FEATURES, Table 1) for documentation, just not fed
# to any model. This is arguably MORE faithful to Novelty 1's own framing
# (R,K,LS,C,P used as "spatially explicit ML input features") than handing
# the model the precomputed answer.
RUSLE_FEATURES = ['R_Factor', 'K_Factor', 'LS_Factor', 'C_Factor', 'P_Factor']
TOPO_FEATURES = ['Elevation', 'Slope', 'Aspect_sin', 'Aspect_cos',
                  'Curvature_Plan', 'Curvature_Profile', 'TWI',
                  'SPI', 'TPI', 'TRI', 'Flow_Accumulation', 'Valley_Depth']
RS_FEATURES = ['NDVI', 'NDWI', 'NDBI', 'EVI', 'BSI',
               'SAVI', 'MSAVI', 'Albedo']
CLIMATE_FEATURES = ['Annual_Rainfall', 'Rainfall_Seasonality',
                     'Max_Monthly_Rain']
SOIL_FEATURES = ['Clay_Content', 'Sand_Content', 'SOC', 'Bulk_Density']
COORD_COLS = ['X_UTM', 'Y_UTM']
TARGET = 'Soil_Loss'
TARGET_CLASS = 'Erosion_Class'

# Raw ~30-feature schema (before Aspect circular transform / engineered
# features). Includes A_RUSLE (reported/documented) even though it's
# excluded from RUSLE_FEATURES (the modeling feature list) -- see the note
# on RUSLE_FEATURES above.
RAW_FEATURES = (RUSLE_FEATURES + ['A_RUSLE', 'Elevation', 'Slope', 'Aspect',
                'Curvature_Plan', 'Curvature_Profile', 'TWI', 'SPI', 'TPI',
                'TRI', 'Flow_Accumulation', 'Valley_Depth'] + RS_FEATURES +
                CLIMATE_FEATURES + SOIL_FEATURES)

# Feature groups keyed by name -> used for CNN-LSTM three-stream model & figures
FEATURE_GROUPS = {
    "RUSLE": RUSLE_FEATURES,
    "Topographic": TOPO_FEATURES,
    "Spectral_Climate_Soil": RS_FEATURES + CLIMATE_FEATURES + SOIL_FEATURES,
}

VIF_THRESHOLD = 10.0        # remove features with VIF > 10

# ── Erosion Classes ────────────────────────────────────────────
EROSION_BINS = [0, 5, 15, 30, 50, 100, float('inf')]
EROSION_LABELS = ['Very Low', 'Low', 'Moderate',
                   'High', 'Very High', 'Extreme']
EROSION_COLORS = ['#1a9641', '#a6d96a', '#ffffbf',
                   '#fdae61', '#d7191c', '#730000']

# ── ML Hyperparameters ─────────────────────────────────────────
RF_PARAMS = {
    'n_estimators': 500, 'max_depth': None,
    'min_samples_split': 4, 'min_samples_leaf': 2,
    'max_features': 'sqrt', 'n_jobs': -1,
    'random_state': 42, 'oob_score': True
}
# GPU toggle: this machine has an NVIDIA GTX 1650 Ti. XGBoost and CatBoost
# both bundle their own CUDA runtime in the pip/conda wheel and use the GPU
# with no extra system setup; TensorFlow (06_dl_models.py) needs the
# separate CUDA 11.2 / cuDNN 8.1 conda env (see README_gpu_setup.md) and
# uses the GPU automatically whenever one is visible -- no per-model flag.
USE_GPU = True

XGB_PARAMS = {
    'n_estimators': 600, 'learning_rate': 0.03,
    'max_depth': 6, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1,
    'reg_lambda': 1.5, 'random_state': 42, 'n_jobs': -1,
    **({'device': 'cuda', 'tree_method': 'hist'} if USE_GPU else {}),
}
GB_PARAMS = {
    'n_estimators': 400, 'learning_rate': 0.05,
    'max_depth': 5, 'subsample': 0.75, 'random_state': 42
}  # sklearn's GradientBoostingRegressor has no GPU implementation
CATBOOST_PARAMS = {
    'iterations': 500, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3,
    'random_state': 42, 'verbose': 0,
    **({'task_type': 'GPU', 'devices': '0'} if USE_GPU else {}),
}
STACKING_META = 'ridge'    # meta-learner: 'ridge' or 'linear'

# Monotone constraints for XGBoost, aligned to the final selected feature
# order at train time (built dynamically in 04_ml_models.py); this dict maps
# feature name -> desired monotonic sign and is looked up there.
XGB_MONOTONE_POSITIVE = ['Slope', 'LS_Factor', 'R_Factor', 'K_Factor']
XGB_MONOTONE_NEGATIVE = ['NDVI', 'C_Factor']

# ── DL Hyperparameters ─────────────────────────────────────────
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE_ES = 25            # EarlyStopping
PATIENCE_LR = 12            # ReduceLROnPlateau
LR_ANN = 0.001
LR_CNN_LSTM = 0.0005
LR_TRANSFORMER = 0.0003
MIN_LR = 1e-6

# ── Uncertainty ────────────────────────────────────────────────
CONFIDENCE_LEVEL = 0.90     # 90% prediction intervals
QUANTILES = [0.05, 0.50, 0.95]

# ── Visualization ──────────────────────────────────────────────
DPI = 300
FIG_FORMATS = ['png', 'pdf', 'svg']
STYLE = 'seaborn-v0_8-whitegrid'
FONT_FAMILY = 'DejaVu Sans'   # Arial not always present on Windows/matplotlib; safe fallback
FONT_SIZE = 11
PALETTE = {
    'Random Forest': '#1565C0',
    'XGBoost': '#2E7D32',
    'Gradient Boosting': '#E65100',
    'CatBoost': '#6A1B9A',
    'Stacking Ensemble': '#B71C1C',
    'ANN-Residual': '#00838F',
    'CNN-LSTM-Attention': '#F57F17',
    'Transformer': '#4E342E',
}
MODEL_ORDER = list(PALETTE.keys())

# ── GeoTIFF Export ─────────────────────────────────────────────
RASTER_RES = 30              # output raster resolution (meters), used for real DEM-derived rasters
NODATA_VALUE = -9999
BATCH_SIZE_RASTER = 10000    # pixels per prediction batch
# When no real DEM/feature_stack.tif exists yet, a synthetic demo raster is
# generated over WATERSHED_BBOX with its long edge capped at this many
# pixels (keeps full-pipeline runs tractable; swap in the real 30 m stack
# once C1's raster stack is available -- no code changes needed elsewhere).
RASTER_DEMO_DIM = 400

# ── Journal Submission ─────────────────────────────────────────
JOURNAL_NAME = "Catena"
MAX_WORDS = 8000
MAX_FIGURES = 16
MAX_TABLES = 8
