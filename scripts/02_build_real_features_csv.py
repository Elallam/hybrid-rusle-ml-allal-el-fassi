"""
02_build_real_features_csv.py -- Build the real
data/features_j1_allalelfassi.csv from the real watershed raster stack.
Thin wrapper around src.real_data.build_training_csv (also called
automatically by 01_data_loading.py if real data is available but the CSV
hasn't been built yet -- this script exists for explicit/manual (re)builds
and to print a summary + provenance note).

See src/real_data.py's module docstring for full per-feature provenance.
Year 2025 chosen as the "current state" epoch for a susceptibility map
(same choice C2 made): mapping one representative epoch rather than
pooling multi-year pixels, which would pseudo-replicate spatially
autocorrelated samples across years.
"""
import os
import sys
import pandas as pd

J1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, J1_DIR)
sys.path.insert(0, os.path.join(J1_DIR, "src"))
import config
from real_data import build_training_csv

YEAR = 2025


def main():
    print(f"Building real watershed layer stack + training CSV (epoch {YEAR})...")
    df = build_training_csv(config.DATA_CSV, year=YEAR)

    readme_path = os.path.join(os.path.dirname(config.DATA_CSV), "README_data_provenance.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(
            f"features_j1_allalelfassi.csv -- REAL data, built {pd.Timestamp.now().date()}.\n"
            f"Source: allal-erosion (C1 RUSLE pipeline) + Comparative Evaluation of\n"
            f"Machine Learning (C2, Sentinel-2 indices) + this project's own GEE fetch\n"
            f"(scripts/01_fetch_j1_extra_layers.py: MSAVI, Albedo, Bulk_Density), epoch {YEAR}.\n"
            f"{len(df)} stratified-random pixel samples across the FAO Soil_Loss classes,\n"
            f"drawn from the watershed-clipped master grid (EPSG:32629, 30m).\n"
            f"See src/real_data.py for full per-feature provenance. A_RUSLE is included\n"
            f"as a documented column but is NOT a model input feature (see config.py's\n"
            f"RUSLE_FEATURES docstring: A_RUSLE equals Soil_Loss by construction on real\n"
            f"data, so it would leak the target).\n"
        )
    print(f"  Saved: {readme_path}")

    print("\nSummary statistics:")
    print(df.describe().T[['min', 'max', 'mean', 'std']].round(3))


if __name__ == "__main__":
    main()
