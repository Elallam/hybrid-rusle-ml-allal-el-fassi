# J1 data provenance

## Current state: REAL DATA (as of 2026-09-16)

`features_j1_allalelfassi.csv` and `feature_stack.tif` are now built from
real GIS data, not the synthetic generator. Source breakdown:

- **C1** (`allal-erosion`): DEM, flow accumulation, LS/K/R/C/P RUSLE
  factors, NDVI, 12-band monthly CHIRPS, 8-band SoilGrids
  (sand/silt/clay/soc at 0-5cm and 5-15cm), A (target, epoch 2025).
- **C2** (`Comparative Evaluation of Machine Learning`): 5-band Sentinel-2
  indices (NDWI, NDBI, EVI, BSI, SAVI), `data/raw/sentinel2_indices_2025.tif`.
- **This project** (`scripts/01_fetch_j1_extra_layers.py`): the only 3
  layers not already available anywhere -- MSAVI, Albedo (Sentinel-2) and
  Bulk_Density (SoilGrids bdod) -- fetched via GEE (project
  `bircool2-c6a44`, run under the `erosion` conda env), saved to
  `data/raw/j1_extra_layers_2025.tif`.
- **Derived locally** (`src/real_data.py`, no fetch needed): Curvature_Plan/
  Curvature_Profile (Zevenbergen & Thorne finite-difference formulas, split
  from C2's combined single Curvature), TPI and Valley_Depth (windowed
  relief from the DEM), Rainfall_Seasonality and Max_Monthly_Rain (CV and
  max of the 12 CHIRPS monthly bands), Sand_Content and SOC (bands 1-2 and
  7-8 of C1's existing 8-band soilgrids.tif, alongside the clay bands C2
  already used).

3000 pixels were stratified-sampled by FAO erosion class from the full
5.26M-pixel valid watershed extent (`src/real_data.build_training_csv`).
`10_geotiff_export.py` builds `feature_stack.tif` the same way, downsampled
(`config.RASTER_DEMO_DIM`, stride-subsampled) for tractable per-pixel
prediction and spatial SHAP interpolation.

## Important: A_RUSLE is excluded from the model features

`A_RUSLE = R_Factor * K_Factor * LS_Factor * C_Factor * P_Factor` is
computed and kept as a **documented** column (Table 1, dataset description)
but is **not** fed to any model. On real data, `Soil_Loss` (the target) IS
`A_2025.tif` -- the exact same product -- so including A_RUSLE as an input
feature would make prediction a trivial identity mapping (R2 approx 1.0),
not genuine learning. See `config.py`'s `RUSLE_FEATURES` docstring. This
was caught and fixed 2026-09-16: the first synthetic-only runs (before this
fix) showed suspiciously high R2 (~0.85-0.89) almost entirely driven by
A_RUSLE dominating SHAP/feature importance; after excluding it, real-data
R2 dropped to a believable ~0.6-0.75 range, consistent with published
erosion-susceptibility ML studies.

## Rebuilding

```
# 1. (only if MSAVI/Albedo/Bulk_Density need refreshing, e.g. a new year)
conda run -n erosion python scripts/01_fetch_j1_extra_layers.py

# 2. Rebuild the training CSV (also happens automatically if the CSV is
#    deleted and 01_data_loading.py finds real data available)
python scripts/02_build_real_features_csv.py
```

Delete `data/features_j1_allalelfassi.csv` and `data/feature_stack.tif` and
rerun `main.py` to regenerate both from the current raster stack.

## Falling back to synthetic

If the sibling C1/C2 projects or the GEE fetch aren't available (e.g. a
fresh clone on another machine), `01_data_loading.py` and
`10_geotiff_export.py` both auto-detect this (`real_data.real_data_available()`)
and fall back to the physically-consistent synthetic generator (Section 3.5
of `CLAUDE_Article_J1.md`) automatically -- no code changes needed either
way.
