"""
13_tables.py -- Remaining publication tables for J1 (CLAUDE.md Section 10).

Table 2 (VIF) is saved by 03_feature_engineering.py and Table 4 (main
performance) by 09_evaluation.py as they're produced as a side effect of
those stages' own computation. This module builds Tables 1, 3, 5, 6, 7, 8.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew

import config
from utils import save_table

FEATURE_META = {}
for f in config.RUSLE_FEATURES:
    FEATURE_META[f] = ('RUSLE factor', 'CHIRPS/SoilGrids/DEM/Sentinel-2/LULC', '30 m', 'dimensionless')
for f in config.TOPO_FEATURES:
    FEATURE_META[f] = ('Topographic', 'SRTM 30 m DEM', '30 m', 'varies')
for f in config.RS_FEATURES:
    FEATURE_META[f] = ('Remote sensing index', 'Sentinel-2 L2A', '10->30 m', 'dimensionless')
for f in config.CLIMATE_FEATURES:
    FEATURE_META[f] = ('Climate', 'CHIRPS/ERA5', '~5 km->30 m', 'mm or %')
for f in config.SOIL_FEATURES:
    FEATURE_META[f] = ('Soil', 'SoilGrids 250 m', '250 m->30 m', '% or g/kg or kg/dm3')


def table01_dataset_description(df):
    rows = []
    cols = [c for c in config.RAW_FEATURES if c in df.columns] + [config.TARGET]
    for c in cols:
        group, source, res, unit = FEATURE_META.get(c, ('Target', 'C1 RUSLE baseline', '30 m', 't/ha/yr'))
        s = df[c].dropna()
        rows.append({'Feature': c, 'Group': group, 'Source': source, 'Resolution': res,
                     'Unit': unit, 'Min': s.min(), 'Max': s.max(), 'Mean': s.mean(),
                     'Std': s.std(), 'Skewness': skew(s)})
    table = pd.DataFrame(rows)
    save_table(table, "table01_dataset_description.csv")
    return table


def table03_hyperparameter_optimization(tuning_table):
    save_table(tuning_table, "table03_hyperparameter_optimization.csv")
    return tuning_table


def table05_spatial_cv_detail(block_cv_table):
    fold_cols = [c for c in block_cv_table.columns if c.startswith('Fold')]
    out = block_cv_table[['Model'] + fold_cols + ['Mean_R2_spatial', 'Std_R2_spatial',
                                                    'Mean_R2_standard', 'Difference']].copy()
    out = out.rename(columns={'Mean_R2_spatial': 'Mean_R2', 'Std_R2_spatial': 'Std_R2',
                               'Mean_R2_standard': 'CV_R2_standard'})
    save_table(out, "table05_spatial_cv_detail.csv")
    return out


def table06_feature_importance_synthesis(importance_synthesis, xgb_importance, features):
    df = importance_synthesis.set_index('Feature').loc[features].reset_index()
    df['XGB_Importance'] = xgb_importance
    for col in ('RF_Importance', 'XGB_Importance', 'SHAP_Mean_Abs'):
        df[f'{col}_rank'] = df[col].rank(ascending=False)
    df['Mean_Rank'] = df[[f'{c}_rank' for c in
                           ('RF_Importance', 'XGB_Importance', 'SHAP_Mean_Abs')]].mean(axis=1)
    df['Group'] = df['Feature'].map(lambda f: FEATURE_META.get(f, ('Engineered', '', '', ''))[0])
    df = df.sort_values('Mean_Rank').reset_index(drop=True)
    df.insert(0, 'Rank', np.arange(1, len(df) + 1))
    out = df[['Rank', 'Feature', 'Group', 'RF_Importance', 'XGB_Importance', 'SHAP_Mean_Abs',
              'In_RFE', 'Mean_Rank', 'Selected_Final']].rename(
        columns={'In_RFE': 'RFE_Selected', 'Selected_Final': 'Selected'})
    save_table(out, "table06_feature_importance_synthesis.csv")
    return out


def table07_uncertainty_results(uncertainty_results, y_test, coords_test):
    from importlib.util import spec_from_file_location, module_from_spec
    import os
    spec = spec_from_file_location("ev13", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "09_evaluation.py"))
    ev = module_from_spec(spec)
    spec.loader.exec_module(ev)

    rows = []
    labels = {'bootstrap': 'Bootstrap Ensemble', 'qrf': 'Quantile Regression Forest',
              'conformal': 'Conformal Prediction'}
    for key, label in labels.items():
        r = uncertainty_results[key]
        mean_pred = r.get('mean', r.get('median'))
        residuals = y_test - mean_pred
        I, _ = ev.morans_i(residuals, coords_test)
        mpiw = float(np.mean(r['upper'] - r['lower']))
        rows.append({'Model': label, 'Coverage_90pct': r['coverage'], 'MPIW': mpiw,
                     'PICP': r['coverage'], 'Moran_I_Residuals': I})
    table = pd.DataFrame(rows)
    save_table(table, "table07_uncertainty_results.csv")
    return table


def table08_literature_comparison(this_study_r2=None, this_study_rmse=None):
    rows = [
        ("Gayen et al.", 2019, "India", "Chahinda catchment", "LR, RF", "Random Forest",
         "TBD", "TBD", "Non-spatial k-fold", "Catena"),
        ("Pham et al.", 2020, "Vietnam", "N/A", "REPTree ensembles", "REPTree+Bagging",
         "TBD", "TBD", "10-fold CV", "Catena"),
        ("Avand et al.", 2021, "Iran", "N/A", "ML flood/erosion models", "TBD",
         "TBD", "TBD", "TBD", "Earth Science Informatics"),
        ("Ouallali et al.", 2020, "Morocco", "Sebou-adjacent", "RUSLE + field sediment yield",
         "RUSLE", "TBD", "TBD", "Field validation", "Catena"),
        ("Benavidez et al.", 2018, "Global review", "N/A", "R/USLE review", "N/A",
         "N/A", "N/A", "N/A", "NHESS"),
        ("This study (J1)", 2026, "Morocco", "Allal El Fassi, Sebou Basin",
         "RF, XGBoost, GB, CatBoost, Stacking, ANN-Res, CNN-LSTM-Attention, Transformer",
         "Stacking Ensemble",
         f"{this_study_r2:.3f}" if this_study_r2 is not None else "TBD",
         f"{this_study_rmse:.2f}" if this_study_rmse is not None else "TBD",
         "SLOOCV + spatial block CV", "Catena (target)"),
    ]
    table = pd.DataFrame(rows, columns=["Reference", "Year", "Country", "Watershed", "Method",
                                         "Best_Model", "R2", "RMSE", "Validation_Type", "Journal"])
    save_table(table, "table08_literature_comparison.csv")
    return table


if __name__ == "__main__":
    pass
