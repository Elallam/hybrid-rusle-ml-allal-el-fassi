"""
02_preprocessing.py -- Enhanced preprocessing pipeline for J1.

CLAUDE.md Section 5, STEPS 2-4:
  STEP 2  Aspect circular transformation
  STEP 3  Missing value analysis (median <5% missing, MICE >5% missing)
  STEP 4  Outlier detection & treatment (target: IQR; features: winsorize)
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest

import config


def aspect_circular_transform(df):
    """Aspect is circular (0deg == 360deg); replace with sin/cos components."""
    if 'Aspect' in df.columns:
        df = df.copy()
        df['Aspect_sin'] = np.sin(np.radians(df['Aspect']))
        df['Aspect_cos'] = np.cos(np.radians(df['Aspect']))
        df = df.drop(columns=['Aspect'])
    return df


def handle_missing_values(df, feature_cols):
    df = df.copy()
    miss_frac = df[feature_cols].isna().mean()
    drop_cols = miss_frac[miss_frac > 0.30].index.tolist()
    if drop_cols:
        print(f"  Dropping features with >30% missing: {drop_cols}")
        feature_cols = [c for c in feature_cols if c not in drop_cols]
        df = df.drop(columns=drop_cols)

    low_missing = [c for c in feature_cols if 0 < miss_frac.get(c, 0) <= 0.05]
    high_missing = [c for c in feature_cols if miss_frac.get(c, 0) > 0.05]

    if low_missing:
        imputer = SimpleImputer(strategy='median')
        df[low_missing] = imputer.fit_transform(df[low_missing])
        print(f"  Median-imputed (<=5% missing): {low_missing}")

    if high_missing:
        imputer = IterativeImputer(random_state=config.RANDOM_STATE, max_iter=10)
        df[high_missing] = imputer.fit_transform(df[high_missing])
        print(f"  MICE-imputed (>5% missing): {high_missing}")

    return df, feature_cols


def treat_target_outliers(df, target_col):
    """IQR rule (3xIQR) on the target -- treat as measurement error, remove."""
    df = df.copy()
    q1, q3 = df[target_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    n_before = len(df)
    df = df[(df[target_col] >= lower) & (df[target_col] <= upper)].reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  Target IQR(3x) outlier removal: {n_removed} rows removed "
          f"(bounds [{lower:.2f}, {upper:.2f}])")
    return df


def treat_feature_outliers(df, feature_cols):
    """Isolation Forest flags outliers; features are winsorized (1st/99th pct),
    never removed -- they may be physically real extreme terrain/climate values."""
    df = df.copy()
    iso = IsolationForest(contamination=0.05, random_state=config.RANDOM_STATE, n_jobs=-1)
    flags = iso.fit_predict(df[feature_cols])
    n_flagged = int((flags == -1).sum())
    print(f"  Isolation Forest flagged {n_flagged} feature-outlier rows "
          f"({100 * n_flagged / len(df):.1f}%) -- winsorizing instead of removing")

    for col in feature_cols:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    return df


def run(df):
    print(f"  n_samples before preprocessing: {len(df)}")
    df = aspect_circular_transform(df)

    feature_cols = [c for c in config.RAW_FEATURES if c != 'Aspect' and c in df.columns]
    feature_cols += [c for c in ('Aspect_sin', 'Aspect_cos') if c in df.columns and c not in feature_cols]

    df, feature_cols = handle_missing_values(df, feature_cols)
    df = treat_target_outliers(df, config.TARGET)
    df = treat_feature_outliers(df, feature_cols)

    print(f"  n_samples after preprocessing : {len(df)}")
    return {'df': df, 'feature_cols': feature_cols}


if __name__ == "__main__":
    pass
