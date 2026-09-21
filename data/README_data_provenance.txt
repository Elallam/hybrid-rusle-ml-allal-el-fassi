features_j1_allalelfassi.csv -- REAL data, built 2026-09-16.
Source: allal-erosion (C1 RUSLE pipeline) + Comparative Evaluation of
Machine Learning (C2, Sentinel-2 indices) + this project's own GEE fetch
(scripts/01_fetch_j1_extra_layers.py: MSAVI, Albedo, Bulk_Density), epoch 2025.
3000 stratified-random pixel samples across the FAO Soil_Loss classes,
drawn from the watershed-clipped master grid (EPSG:32629, 30m).
See src/real_data.py for full per-feature provenance. LS_Factor capped at
the 99th percentile (43.82) to tame known FlowAcc^0.4 spikes along
drainage channels. A_RUSLE is included as a documented column but is NOT a
model input feature (see config.py's RUSLE_FEATURES docstring: A_RUSLE
equals Soil_Loss by construction on real data, so it would leak the target).
