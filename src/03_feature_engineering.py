"""
03_feature_engineering.py -- Feature engineering, VIF, selection & splitting.

CLAUDE.md Section 5, STEPS 5-10:
  STEP 5  New engineered features (RUSLE interactions, topo/spectral compounds)
  STEP 6  Variance Inflation Factor analysis (drop VIF > 10)
  STEP 7  Feature selection: RF importance, RFE, SHAP -- compare & merge
  STEP 8  Train/Val/Test split (60/20/20), stratified by erosion class
  STEP 9  Feature scaling (StandardScaler for ML, MinMaxScaler for DL)
  STEP 10 Data augmentation for DL if n_samples < 2000 (Gaussian noise, 2x)
"""
import os
import joblib
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap

import config
from utils import save_table, erosion_class


def engineer_features(df):
    df = df.copy()
    df['R_K_interaction'] = df['R_Factor'] * df['K_Factor']
    df['LS_C_interaction'] = df['LS_Factor'] * df['C_Factor']
    df['Erosivity_index'] = df['R_Factor'] * df['K_Factor'] * df['LS_Factor']

    slope_rad = np.radians(df['Slope'].clip(lower=0.01))
    df['STI'] = (df['Flow_Accumulation'].clip(lower=0.01) / 22.13) ** 0.6 * \
                (np.sin(slope_rad) / 0.0896) ** 1.3
    df['CI'] = -np.log(df['TWI'].abs() / (df['Slope'] + 0.001) + 1e-6)

    df['Modified_BSI'] = (df['NDBI'] - df['NDVI']) / (df['NDBI'] + df['NDVI'] + 1e-6)
    df['Vegetation_fraction'] = 1 - df['C_Factor']

    return df


ENGINEERED_FEATURES = ['R_K_interaction', 'LS_C_interaction', 'Erosivity_index',
                        'STI', 'CI', 'Modified_BSI', 'Vegetation_fraction']


def compute_vif(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    vif_rows = []
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.inf
        vif_rows.append({'Feature': col, 'VIF': vif})
    vif_df = pd.DataFrame(vif_rows)
    vif_df['Status'] = np.where(vif_df['VIF'] > config.VIF_THRESHOLD, 'Remove',
                                 np.where(vif_df['VIF'] > 5, 'Keep (moderate)', 'Keep'))
    vif_df['Reason'] = np.where(vif_df['VIF'] > config.VIF_THRESHOLD,
                                 'Severe multicollinearity (VIF>10)', '')
    vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif_df


def select_features(df, candidate_cols, target_col):
    X = df[candidate_cols].replace([np.inf, -np.inf], np.nan).fillna(df[candidate_cols].median())
    y = df[target_col]

    # Method 1: RF importance
    rf = RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=candidate_cols)
    method1 = set(rf_importance[rf_importance > 0.01].index)

    # Method 2: RFE with CV (on a subsample of estimators for speed)
    rf_small = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    rfe = RFECV(rf_small, step=1, cv=KFold(3, shuffle=True, random_state=config.RANDOM_STATE),
                min_features_to_select=max(5, len(candidate_cols) // 2), n_jobs=-1)
    rfe.fit(X, y)
    method2 = set(np.array(candidate_cols)[rfe.support_])

    # Method 3: SHAP-based selection
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X.sample(min(500, len(X)), random_state=config.RANDOM_STATE))
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=candidate_cols)
    method3 = set(mean_abs_shap[mean_abs_shap > mean_abs_shap.mean() * 0.5].index)

    selected = sorted(method1 | method2 | method3)  # union: keep anything any method values
    synthesis = pd.DataFrame({
        'Feature': candidate_cols,
        'RF_Importance': rf_importance.values,
        'In_RFE': [c in method2 for c in candidate_cols],
        'SHAP_Mean_Abs': mean_abs_shap.values,
        'Selected_RF': [c in method1 for c in candidate_cols],
        'Selected_SHAP': [c in method3 for c in candidate_cols],
        'Selected_Final': [c in selected for c in candidate_cols],
    }).sort_values('RF_Importance', ascending=False).reset_index(drop=True)

    print(f"  RF-importance selected : {len(method1)} features")
    print(f"  RFE-CV selected         : {len(method2)} features")
    print(f"  SHAP-based selected     : {len(method3)} features")
    print(f"  Final (union) feature set: {len(selected)} features")

    return selected, synthesis


def augment_with_noise(X, y, factor=2, sigma_scale=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    std = X.std(axis=0)
    X_aug, y_aug = [X], [y]
    for _ in range(factor - 1):
        noise = rng.normal(0, sigma_scale * std, size=X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)


def safe_strat_labels(strat, min_count=15):
    """Merge any class with too few members into its nearest-value neighbour
    so the 3-way stratified split below never crashes on a rare erosion
    class (sklearn requires >=2 members per class per split; two sequential
    splits need a bit more headroom than that). Only affects the temporary
    labels used to STRATIFY the split -- the real Erosion_Class column
    (and everything derived from it) is untouched."""
    s = strat.copy()
    while True:
        counts = s.value_counts()
        small = counts[counts < min_count]
        if small.empty or len(counts) <= 1:
            break
        label = small.index[0]
        remaining = counts.index[counts.index != label]
        nearest = min(remaining, key=lambda l: abs(l - label))
        s = s.replace(label, nearest)
    return s


def run(df):
    df = engineer_features(df)

    candidate_cols = [c for c in (config.RUSLE_FEATURES + config.TOPO_FEATURES +
                                   config.RS_FEATURES + config.CLIMATE_FEATURES +
                                   config.SOIL_FEATURES + ENGINEERED_FEATURES)
                       if c in df.columns]

    # Three-way split, stratified by erosion class -- done BEFORE VIF/feature
    # selection so neither sees the val/test rows (avoids selection leakage).
    raw_strat = df[config.TARGET_CLASS] if config.TARGET_CLASS in df.columns else erosion_class(df[config.TARGET])
    strat = safe_strat_labels(raw_strat)
    df_train, df_temp = train_test_split(
        df, train_size=config.TRAIN_RATIO, stratify=strat, random_state=config.RANDOM_STATE)
    val_share_of_temp = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    strat_temp = strat.loc[df_temp.index]
    df_val, df_test = train_test_split(
        df_temp, train_size=val_share_of_temp, stratify=strat_temp, random_state=config.RANDOM_STATE)

    print(f"  Split sizes -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    vif_table = compute_vif(df_train, candidate_cols)
    save_table(vif_table, "table02_vif_analysis.csv")
    keep_cols = vif_table.loc[vif_table['Status'] != 'Remove', 'Feature'].tolist()
    print(f"  VIF: kept {len(keep_cols)}/{len(candidate_cols)} features (removed VIF>{config.VIF_THRESHOLD})")

    selected_features, importance_synthesis = select_features(df_train, keep_cols, config.TARGET)

    X_train = df_train[selected_features].to_numpy(dtype=float)
    X_val = df_val[selected_features].to_numpy(dtype=float)
    X_test = df_test[selected_features].to_numpy(dtype=float)
    y_train = df_train[config.TARGET].to_numpy(dtype=float)
    y_val = df_val[config.TARGET].to_numpy(dtype=float)
    y_test = df_test[config.TARGET].to_numpy(dtype=float)
    coords_train = df_train[config.COORD_COLS].to_numpy(dtype=float)
    coords_val = df_val[config.COORD_COLS].to_numpy(dtype=float)
    coords_test = df_test[config.COORD_COLS].to_numpy(dtype=float)
    class_train = df_train[config.TARGET_CLASS].to_numpy(dtype=int)
    class_val = df_val[config.TARGET_CLASS].to_numpy(dtype=int)
    class_test = df_test[config.TARGET_CLASS].to_numpy(dtype=int)

    scaler_standard = StandardScaler().fit(X_train)
    scaler_minmax = MinMaxScaler().fit(X_train)
    joblib.dump(scaler_standard, os.path.join(config.MODELS_DIR, "scaler_standard.pkl"))
    joblib.dump(scaler_minmax, os.path.join(config.MODELS_DIR, "scaler_minmax.pkl"))

    X_train_std = scaler_standard.transform(X_train)
    X_val_std = scaler_standard.transform(X_val)
    X_test_std = scaler_standard.transform(X_test)
    X_train_mm = scaler_minmax.transform(X_train)
    X_val_mm = scaler_minmax.transform(X_val)
    X_test_mm = scaler_minmax.transform(X_test)

    if len(X_train) < 2000:
        X_train_mm_dl, y_train_dl = augment_with_noise(
            X_train_mm, y_train, factor=2, random_state=config.RANDOM_STATE)
        print(f"  DL data augmentation applied: {len(X_train)} -> {len(X_train_mm_dl)} training samples")
    else:
        X_train_mm_dl, y_train_dl = X_train_mm, y_train

    return {
        'features': selected_features,
        'vif_table': vif_table,
        'importance_synthesis': importance_synthesis,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_std': X_train_std, 'X_val_std': X_val_std, 'X_test_std': X_test_std,
        'X_train_mm': X_train_mm, 'X_val_mm': X_val_mm, 'X_test_mm': X_test_mm,
        'X_train_mm_dl': X_train_mm_dl, 'y_train_dl': y_train_dl,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'coords_train': coords_train, 'coords_val': coords_val, 'coords_test': coords_test,
        'class_train': class_train, 'class_val': class_val, 'class_test': class_test,
        'scaler_standard': scaler_standard, 'scaler_minmax': scaler_minmax,
        'df_engineered': df,
    }


if __name__ == "__main__":
    pass
