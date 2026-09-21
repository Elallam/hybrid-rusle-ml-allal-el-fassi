"""
main.py -- Article J1: Hybrid AI-RUSLE Framework for Soil Erosion Susceptibility
Run: python main.py [--mode full|quick|predict_only]

STEP  1  -> Load and validate data + coordinates
STEP  2  -> Preprocessing (clean, transform, aspect fix)
STEP  3  -> Feature engineering (VIF, interactions, selection, split, scale)
STEP  4  -> Train ML models (RF, XGB, GB, CatBoost)
STEP  5  -> Train Stacking Ensemble (J1 core novelty) + ablation study
STEP  6  -> Train DL models (ANN-Res, CNN-LSTM-Attention, Transformer)
STEP  7  -> Spatial LOOCV (RF + XGB)
STEP  8  -> Spatial block CV (all tree models + stacking)
STEP  9  -> Uncertainty quantification (Bootstrap + QRF + Conformal)
STEP 10  -> Compute all metrics (Table 4)
STEP 11  -> SHAP analysis (standard + spatial SHAP maps)
STEP 12  -> GeoTIFF export (prediction + classification + uncertainty rasters)
STEP 13  -> Generate all 16 figures and 8 tables
STEP 14  -> Auto-generate Results section draft
STEP 15  -> Print final summary report
"""
import os
import sys
import argparse
import importlib.util
import time

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

import config
from utils import print_step, ensure_dirs, save_table

TOTAL_STEPS = 15


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_quick_mode():
    """Shrink every expensive knob so the full pipeline runs in minutes, not
    hours -- for smoke-testing the pipeline end to end before a full run."""
    config.N_SAMPLES_SYN = 600
    config.RF_PARAMS = {**config.RF_PARAMS, 'n_estimators': 120}
    config.XGB_PARAMS = {**config.XGB_PARAMS, 'n_estimators': 120}
    config.GB_PARAMS = {**config.GB_PARAMS, 'n_estimators': 80}
    config.CATBOOST_PARAMS = {**config.CATBOOST_PARAMS, 'iterations': 120}
    config.EPOCHS = 15
    config.PATIENCE_ES = 6
    config.PATIENCE_LR = 3
    config.N_BOOTSTRAP = 15
    config.SLOOCV_MAX_POINTS = 30
    config.N_FOLDS_BLOCK = 3
    config.RASTER_DEMO_DIM = 80
    print("  [quick mode] reduced sample size, estimator counts, epochs, "
          "bootstrap iterations, SLOOCV points and raster resolution.")


def make_predict_fn(model_name, model, group_idx):
    """DL models need array-shape adjustments (multi-input for CNN-LSTM,
    keras verbose flag) that plain sklearn-style model.predict() doesn't
    handle -- everything else (RF/XGB/GB/CatBoost/Stacking) uses model.predict directly."""
    if model_name == 'CNN-LSTM-Attention':
        def _fn(X):
            groups = [X[:, idx][:, :, None] for idx in group_idx.values()]
            return model.predict(groups, verbose=0).flatten()
        return _fn
    if model_name in ('ANN-Residual', 'Transformer'):
        def _fn(X):
            return model.predict(X, verbose=0).flatten()
        return _fn
    return None


def print_summary(table4, n_figures, n_tables, best_model_name):
    best = table4.iloc[0]
    lines = [
        "ARTICLE J1 -- HYBRID AI-RUSLE FRAMEWORK -- RESULTS SUMMARY",
        f"Best model     : {best_model_name} R2={best['R2']:.4f} RMSE={best['RMSE']:.2f} t/ha/yr",
        f"NSE / KGE / IOA: {best['NSE']:.3f} / {best['KGE']:.3f} / {best['IOA']:.3f}",
        f"Figures saved  : {n_figures} files (PNG+PDF+SVG)",
        f"Tables  saved  : {n_tables} CSV files",
        f"Models  saved  : {config.MODELS_DIR}",
        f"Rasters saved  : {config.RASTERS_DIR}",
    ]
    width = max(len(l) for l in lines) + 2
    print("\n" + "+" + "-" * width + "+")
    for i, l in enumerate(lines):
        pad = width - len(l)
        print("| " + l + " " * (pad - 1) + "|")
        if i == 0:
            print("+" + "-" * width + "+")
    print("+" + "-" * width + "+")


def main():
    parser = argparse.ArgumentParser(description="Article J1 hybrid AI-RUSLE pipeline")
    parser.add_argument("--mode", choices=["full", "quick", "predict_only"], default="full")
    args = parser.parse_args()

    ensure_dirs()
    np.random.seed(config.RANDOM_STATE)
    t_start = time.time()

    if args.mode == "quick":
        apply_quick_mode()
    elif args.mode == "predict_only":
        raise NotImplementedError(
            "--mode predict_only is not wired up yet (would need to reload every "
            "saved model + scaler from outputs/models/ and skip straight to Step 12's "
            "GeoTIFF export). Use --mode full or --mode quick, which always retrain.")

    dl_mod = _load("dl06", "01_data_loading.py")
    pp_mod = _load("pp02", "02_preprocessing.py")
    fe_mod = _load("fe03", "03_feature_engineering.py")
    ml_mod = _load("ml04", "04_ml_models.py")
    stk_mod = _load("stk05", "05_stacking_ensemble.py")
    dlm_mod = _load("dlm06", "06_dl_models.py")
    scv_mod = _load("scv08", "08_spatial_cv.py")
    unc_mod = _load("unc07", "07_uncertainty.py")
    ev_mod = _load("ev09", "09_evaluation.py")
    shap_mod = _load("shap11", "11_shap_analysis.py")
    geo_mod = _load("geo10", "10_geotiff_export.py")
    fig_mod = _load("fig12", "12_figures.py")
    tab_mod = _load("tab13", "13_tables.py")
    paper_mod = _load("paper14", "14_paper_writing_aid.py")

    print_step(1, TOTAL_STEPS, "Loading and validating data")
    data = dl_mod.run()

    print_step(2, TOTAL_STEPS, "Preprocessing (aspect transform, missing values, outliers)")
    pp_out = pp_mod.run(data['df'])

    print_step(3, TOTAL_STEPS, "Feature engineering (VIF, selection, split, scaling)")
    fe = fe_mod.run(pp_out['df'])
    features = fe['features']
    print(f"  Final feature set ({len(features)}): {features}")

    print_step(4, TOTAL_STEPS, "Training base ML models (RF, XGBoost, GB, CatBoost)")
    ml_results, tuning_table = ml_mod.train_all(
        features, fe['X_train_std'], fe['y_train'], fe['X_test_std'], tune=(args.mode != "quick"))

    print_step(5, TOTAL_STEPS, "Training Stacking Ensemble (J1 core novelty) + ablation")
    stack_result = stk_mod.build_stacking(
        features, fe['X_train_std'], fe['y_train'], fe['X_test_std'], fe['y_test'])
    from sklearn.metrics import r2_score
    ablation_table = stk_mod.ablation_study(
        features, fe['X_train_std'], fe['y_train'], fe['X_test_std'], fe['y_test'],
        r2_score(fe['y_test'], stack_result['y_pred']))
    save_table(ablation_table, "table_stacking_ablation.csv")

    print_step(6, TOTAL_STEPS, "Training DL models (ANN-Residual, CNN-LSTM-Attention, Transformer)")
    dl_results, group_idx = dlm_mod.train_all(
        features, fe['X_train_mm_dl'], fe['y_train_dl'], fe['X_val_mm'], fe['y_val'],
        fe['X_test_mm'], fe['y_test'])

    all_results = {**ml_results, 'Stacking Ensemble': stack_result, **dl_results}

    print_step(7, TOTAL_STEPS, "Spatial Leave-One-Out CV (RF + XGBoost)")
    sloocv_results = scv_mod.run_sloocv(features, fe['X_train_std'], fe['y_train'], fe['coords_train'])

    print_step(8, TOTAL_STEPS, "Spatial block CV vs standard CV (all base models + stacking)")
    block_cv_table = scv_mod.run_block_cv(features, fe['X_train_std'], fe['y_train'], fe['coords_train'])

    print_step(9, TOTAL_STEPS, "Uncertainty quantification (Bootstrap, QRF, Conformal)")
    uncertainty_results = unc_mod.run(fe['X_train_std'], fe['y_train'], fe['X_test_std'], fe['y_test'])

    print_step(10, TOTAL_STEPS, "Computing full evaluation metric suite (Table 4)")
    table4 = ev_mod.run(all_results, fe['y_test'], features, class_test=fe['class_test'],
                         coords_test=fe['coords_test'], sloocv_results=sloocv_results)
    best_model_name = table4.iloc[0]['Model']
    best_model = all_results[best_model_name]['model']
    print(f"  Best model: {best_model_name} (R2={table4.iloc[0]['R2']:.4f})")

    print_step(11, TOTAL_STEPS, "SHAP analysis (standard + spatial, XGBoost vs Stacking)")
    shap_out = shap_mod.run(ml_results['XGBoost']['model'], stack_result['model'],
                             fe['X_test_std'], features, fe['coords_test'],
                             y_pred_for_repr=stack_result['y_pred'])

    print_step(12, TOTAL_STEPS, "GeoTIFF export (prediction, classification, uncertainty rasters)")
    predict_fn = make_predict_fn(best_model_name, best_model, group_idx)
    scaler_for_best = fe['scaler_minmax'] if best_model_name in dl_results else fe['scaler_standard']
    pred_raster, transform, shape = geo_mod.predict_raster(
        best_model, features, scaler_for_best,
        os.path.join(config.RASTERS_DIR, "soil_loss_predicted.tif"), predict_fn=predict_fn)
    erosion_class_raster = geo_mod.classify_erosion_raster(
        pred_raster, transform, os.path.join(config.RASTERS_DIR, "erosion_class.tif"))
    raster_stats, class_stats_df = geo_mod.raster_statistics(
        erosion_class_raster, transform=transform, is_class=True)
    save_table(class_stats_df, "erosion_class_area_statistics.csv")

    prep_std = geo_mod.prepare_raster_matrix(features, fe['scaler_standard'])
    boot_grid = unc_mod.bootstrap_predict_raster(uncertainty_results['bootstrap']['models'],
                                                  prep_std['X_scaled'])
    uncertainty_rasters = geo_mod.export_uncertainty_rasters(
        boot_grid['mean'], boot_grid['lower'], boot_grid['upper'],
        prep_std['valid_mask'], prep_std['shape'], prep_std['transform'])

    arr_full, transform_full, crs_full, shape_full, descriptions_full = geo_mod.load_raster_stack(config.RASTER_STACK)
    band_order_full = (list(descriptions_full) if descriptions_full and descriptions_full[0]
                        else config.RAW_FEATURES[:arr_full.shape[0]])
    bands_dict_full = {name: arr_full[i] for i, name in enumerate(band_order_full)}
    from rasterio.transform import array_bounds
    left, bottom, right, top = array_bounds(shape_full[0], shape_full[1], transform_full)

    print_step(13, TOTAL_STEPS, "Generating all 16 figures and 8 tables")
    table3 = tab_mod.table03_hyperparameter_optimization(tuning_table)
    table1 = tab_mod.table01_dataset_description(fe['df_engineered'])
    table5 = tab_mod.table05_spatial_cv_detail(block_cv_table)
    xgb_importance = pd.Series(ml_results['XGBoost']['model'].feature_importances_, index=features)
    table6 = tab_mod.table06_feature_importance_synthesis(fe['importance_synthesis'], xgb_importance, features)
    table7 = tab_mod.table07_uncertainty_results(uncertainty_results, fe['y_test'], fe['coords_test'])
    table8 = tab_mod.table08_literature_comparison(
        this_study_r2=table4.iloc[0]['R2'], this_study_rmse=table4.iloc[0]['RMSE'])

    artifacts = {
        'coords': data['coords'], 'raster_bands': bands_dict_full,
        'raster_extent': (left, right, bottom, top),
        'df_engineered': fe['df_engineered'], 'features': features,
        'vif_table': fe['vif_table'], 'table4': table4, 'block_cv_table': block_cv_table,
        'all_results': all_results, 'y_test': fe['y_test'], 'X_test': fe['X_test_std'],
        'shap_xgb': shap_out['shap_xgb'], 'shap_stack': shap_out['shap_stack'],
        'X_test_kernel': shap_out['X_test_kernel'],
        'shap_rasters': shap_out['shap_rasters'], 'top_shap_features': shap_out['top_features'],
        'uncertainty_rasters': uncertainty_rasters, 'erosion_class_raster': erosion_class_raster,
        'class_stats_df': class_stats_df, 'literature_table': table8,
        'best_r2': table4.iloc[0]['R2'],
    }
    fig_mod.run(artifacts)

    print_step(14, TOTAL_STEPS, "Auto-generating Results section draft + cover letter")
    results_text = paper_mod.generate_results_section(
        table4, sloocv_results, table7, shap_out['top_features'], class_stats_df)
    abstract_text = paper_mod.generate_abstract(table4)
    discussion_points = paper_mod.generate_discussion_points(table4, table8)

    draft_path = os.path.join(config.PAPER_DIR, "J1_manuscript_draft.md")
    with open(draft_path, "w", encoding="utf-8") as f:
        f.write("# J1 Manuscript Draft (auto-generated from this run's results)\n\n")
        f.write("## Abstract\n\n" + abstract_text + "\n\n")
        f.write("## Results\n\n" + results_text + "\n\n")
        f.write("## Discussion points\n\n" + "\n".join(f"- {p}" for p in discussion_points) + "\n")
    print(f"  Saved: {draft_path}")

    print_step(15, TOTAL_STEPS, "Final summary report")
    n_figures = len([f for f in os.listdir(config.FIGURES_DIR) if f.endswith('.png')])
    n_tables = len([f for f in os.listdir(config.TABLES_DIR) if f.endswith('.csv')])
    print_summary(table4, n_figures, n_tables, best_model_name)
    print(f"\nTotal runtime: {(time.time() - t_start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
