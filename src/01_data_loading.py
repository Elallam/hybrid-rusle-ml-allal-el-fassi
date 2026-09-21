"""
01_data_loading.py -- Load, validate and report on the J1 feature dataset.

CLAUDE.md Section 5, STEP 1.

If data/features_j1_allalelfassi.csv does not exist yet: real GIS data
(C1 + C2 rasters + this project's own MSAVI/Albedo/Bulk_Density GEE fetch,
see data/README_data.md) is used automatically if available
(real_data.real_data_available()); otherwise a physically consistent
synthetic dataset is generated (Section 3.5) so the pipeline is always
runnable end to end regardless of which machine it's running on.
"""
import os
import numpy as np
import pandas as pd

import config
from utils import generate_synthetic_data, confirm_saved


REQUIRED_COLUMNS = (config.COORD_COLS + config.RAW_FEATURES +
                     [config.TARGET, config.TARGET_CLASS])


def load_or_generate():
    if os.path.exists(config.DATA_CSV):
        df = pd.read_csv(config.DATA_CSV)
        source = "loaded from disk (data/README_data_provenance.txt says which)"
        return df, source

    try:
        from real_data import real_data_available, build_training_csv
        has_real = real_data_available()
    except ImportError:
        has_real = False

    if has_real:
        print(f"  {config.DATA_CSV} not found -- real GIS data available, building "
              f"real training CSV (n={config.N_SAMPLES_SYN}) from the watershed raster stack.")
        df = build_training_csv(config.DATA_CSV, n_samples=config.N_SAMPLES_SYN)
        return df, "real (built from GIS raster stack this run)"

    print(f"  {config.DATA_CSV} not found -- generating synthetic dataset "
          f"(n={config.N_SAMPLES_SYN}) per CLAUDE.md Section 3.5.")
    df = generate_synthetic_data(n_samples=config.N_SAMPLES_SYN,
                                  noise_level=0.12, spatial_structure=True,
                                  random_state=config.RANDOM_STATE)
    df.to_csv(config.DATA_CSV, index=False)
    confirm_saved(config.DATA_CSV)
    return df, "synthetic (generated this run)"


def validate(df):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    non_numeric = [c for c in REQUIRED_COLUMNS
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric dtype in required columns: {non_numeric}")


def report(df, source):
    print(f"  Data source        : {source}")
    print(f"  n_samples           : {len(df)}")
    print(f"  n_raw_features      : {len(config.RAW_FEATURES)}")
    print(f"  Target range (t/ha/yr): [{df[config.TARGET].min():.2f}, "
          f"{df[config.TARGET].max():.2f}], mean={df[config.TARGET].mean():.2f}")
    miss_pct = df[config.RAW_FEATURES].isna().mean().mean() * 100
    print(f"  Mean feature missingness: {miss_pct:.2f}%")
    print(f"  Spatial extent X_UTM: [{df['X_UTM'].min():.0f}, {df['X_UTM'].max():.0f}]")
    print(f"  Spatial extent Y_UTM: [{df['Y_UTM'].min():.0f}, {df['Y_UTM'].max():.0f}]")
    class_counts = df[config.TARGET_CLASS].value_counts().sort_index()
    print("  Erosion class distribution:")
    for cls_id, cnt in class_counts.items():
        label = config.EROSION_LABELS[int(cls_id) - 1]
        print(f"    {cls_id} ({label}): {cnt} ({100 * cnt / len(df):.1f}%)")


def run():
    df, source = load_or_generate()
    validate(df)
    report(df, source)
    coords = df[config.COORD_COLS].copy()
    return {'df': df, 'coords': coords, 'source': source}


if __name__ == "__main__":
    run()
