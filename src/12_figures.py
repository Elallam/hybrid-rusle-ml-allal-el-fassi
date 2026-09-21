"""
12_figures.py -- All 16 publication figures for J1 (CLAUDE.md Section 9).

Every figure is saved as PNG + PDF + SVG at 300 DPI into outputs/figures/.
Where a figure needs real GIS rasters that this project does not have yet
(study-area basemap, RUSLE factor maps), it renders from the same synthetic
demo raster stack used by 10_geotiff_export.py -- swap in real data and
these figures render unchanged.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import config
from utils import set_plot_style, save_fig


# ---------------------------------------------------------------------------
# Small schematic helpers
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, color='#E3F2FD', fontsize=9):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=1.2, edgecolor='#333333', facecolor=color)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fontsize, wrap=True)


def _arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                           linewidth=1.2, color='#333333')
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
def fig01_conceptual_framework():
    fig, ax = plt.subplots(figsize=(17 / 2.54, 10 / 2.54))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    stages = [
        (0.2, 4.6, "Multi-source Data\n(RUSLE R,K,LS,C,P + Topo\n+ RS + Climate + Soil)", '#E3F2FD'),
        (2.7, 4.6, "Preprocessing\n(missing, outliers,\nAspect transform)", '#E8F5E9'),
        (5.2, 4.6, "Feature Engineering\n(interactions, VIF,\nselection)", '#FFF3E0'),
        (7.7, 4.6, "Train/Val/Test Split\n(spatial stratified)", '#F3E5F5'),
        (0.2, 2.6, "Base ML: RF, XGBoost,\nGB, CatBoost", '#E1F5FE'),
        (2.7, 2.6, "Deep Learning: ANN-Res,\nCNN-LSTM-Attention,\nTransformer", '#E1F5FE'),
        (5.2, 2.6, "Stacking Ensemble\n(Ridge meta-learner)\n<< J1 NOVELTY >>", '#FFEBEE'),
        (7.7, 2.6, "SLOOCV + Spatial\nBlock CV validation", '#F1F8E9'),
        (2.7, 0.6, "Uncertainty Quantification\n(Bootstrap, QRF, Conformal)", '#FCE4EC'),
        (5.2, 0.6, "Spatial SHAP Attribution\n(GeoTIFF maps)", '#FCE4EC'),
        (7.7, 0.6, "Erosion Susceptibility Map\n+ Prediction Intervals", '#FFF9C4'),
    ]
    for x, y, text, color in stages:
        _box(ax, (x, y), 2.1, 1.1, text, color, fontsize=7.5)

    flow = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 6), (6, 5), (5, 4),
            (4, 8), (6, 9), (7, 10), (9, 10)]
    centers = [(x + 1.05, y + 0.55) for x, y, *_ in stages]
    for a, b in flow:
        _arrow(ax, centers[a], centers[b])

    ax.set_title("Figure 1. Conceptual framework of the hybrid physical-AI erosion pipeline",
                  fontsize=10)
    save_fig(fig, "fig01_conceptual_framework")
    plt.close(fig)


def fig02_study_area(coords):
    fig, axes = plt.subplots(1, 1, figsize=(14 / 2.54, 12 / 2.54))
    axes.scatter(coords['X_UTM'], coords['Y_UTM'], s=3, alpha=0.4, c='#2E7D32')
    axes.set_xlabel("Easting (m, UTM 29N)"); axes.set_ylabel("Northing (m, UTM 29N)")
    axes.set_title("Figure 2. Study area -- Allal El Fassi sub-watershed sample distribution\n"
                    "(Sebou Basin, Morocco; EPSG:32629)", fontsize=9)
    axes.set_aspect('equal')
    save_fig(fig, "fig02_study_area")
    plt.close(fig)


def fig03_rusle_factor_maps(bands_dict, extent):
    factors = ['R_Factor', 'K_Factor', 'LS_Factor', 'C_Factor', 'P_Factor', 'A_RUSLE']
    fig, axes = plt.subplots(2, 3, figsize=(17 / 2.54, 11 / 2.54))
    for ax, f in zip(axes.flat, factors):
        arr = bands_dict.get(f)
        if arr is None:
            ax.axis('off'); continue
        cmap = 'RdYlGn_r' if f == 'A_RUSLE' else 'viridis'
        im = ax.imshow(arr, cmap=cmap, extent=extent)
        ax.set_title(f, fontsize=9); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 3. RUSLE factor maps", fontsize=10)
    save_fig(fig, "fig03_rusle_factor_maps")
    plt.close(fig)


def fig04_feature_analysis(df, features):
    fig, axes = plt.subplots(1, 2, figsize=(17 / 2.54, 8 / 2.54))
    norm = (df[features] - df[features].mean()) / df[features].std(ddof=0)
    sns.violinplot(data=norm.iloc[:, :min(15, len(features))], ax=axes[0], orient='h',
                    palette='crest')
    axes[0].set_title("A. Feature distributions (z-score)", fontsize=9)

    corr = df[features].corr(method='spearman')
    sns.heatmap(corr, ax=axes[1], cmap='RdBu_r', vmin=-1, vmax=1, cbar_kws={'shrink': 0.6},
                xticklabels=False, yticklabels=False)
    axes[1].set_title("B. Spearman correlation heatmap", fontsize=9)
    fig.suptitle("Figure 4. Feature distribution and correlation analysis", fontsize=10)
    save_fig(fig, "fig04_feature_analysis")
    plt.close(fig)


def fig05_vif_feature_selection(vif_table):
    fig, ax = plt.subplots(figsize=(14 / 2.54, max(8, len(vif_table) * 0.28) / 2.54))
    colors = np.where(vif_table['VIF'] > 10, '#d7191c',
                       np.where(vif_table['VIF'] > 5, '#fdae61', '#1a9641'))
    order = vif_table.sort_values('VIF')
    ax.barh(order['Feature'], order['VIF'].clip(upper=order['VIF'].replace(np.inf, 50).max()),
            color=colors[order.index])
    ax.axvline(5, ls='--', c='orange', lw=1)
    ax.axvline(10, ls='--', c='red', lw=1)
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("Figure 5. VIF analysis and feature selection", fontsize=10)
    save_fig(fig, "fig05_vif_feature_selection")
    plt.close(fig)


def fig06_dl_architecture():
    fig, ax = plt.subplots(figsize=(17 / 2.54, 11 / 2.54))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

    streams = [("RUSLE\n(6 feat)", 0.3, '#E3F2FD'), ("Topographic\n(11 feat)", 3.65, '#E8F5E9'),
               ("Spectral/Climate/Soil\n(13 feat)", 7.0, '#FFF3E0')]
    layer_labels = ["Input", "Conv1D(64,k=3)", "MaxPool1D", "LSTM(64)", "Dense(32)"]
    for label, x0, color in streams:
        for li, ltext in enumerate(layer_labels):
            y = 6.6 - li * 1.1
            _box(ax, (x0, y), 2.6, 0.8, ltext if li else label, color, fontsize=7.5)
            if li > 0:
                _arrow(ax, (x0 + 1.3, y + 1.1), (x0 + 1.3, y + 0.8))

    _box(ax, (3.65, 1.0), 2.6, 0.7, "Stack -> (3, 32)\nMultiHeadAttention(4 heads)", '#FFEBEE', 7)
    for x0 in (0.3 + 1.3, 3.65 + 1.3, 7.0 + 1.3):
        _arrow(ax, (x0, 1.0 + 0.35), (3.65 + 1.3, 1.0 + 0.35))
    _box(ax, (3.65, 0.0), 2.6, 0.7, "Dense(64)->BN->Dropout\nDense(32)->Dense(1)", '#FFF9C4', 7)
    _arrow(ax, (3.65 + 1.3, 1.0), (3.65 + 1.3, 0.7))

    ax.set_title("Figure 6. Three-stream CNN-LSTM-Attention architecture", fontsize=10)
    save_fig(fig, "fig06_dl_architecture")
    plt.close(fig)


def fig07_model_comparison(table4):
    metrics = ['R2', 'NSE', 'KGE', 'IOA']
    models = table4['Model'].tolist()
    fig = plt.figure(figsize=(17 / 2.54, 9 / 2.54))

    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for _, row in table4.iterrows():
        vals = [max(0, row[m]) for m in metrics]
        vals += vals[:1]
        ax1.plot(angles, vals, linewidth=1, label=row['Model'],
                 color=config.PALETTE.get(row['Model']))
    ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(metrics, fontsize=7)
    ax1.set_title("A. Multi-metric radar", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(len(models))
    best = table4['R2'].idxmax()
    colors = [config.PALETTE.get(m, '#999999') for m in models]
    bars = ax2.bar(x, table4['R2'], color=colors)
    bars[best].set_edgecolor('gold'); bars[best].set_linewidth(2.5)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel("R2"); ax2.set_title("B. Model R2 comparison (best model gold-outlined)", fontsize=9)

    fig.suptitle("Figure 7. Model performance comparison", fontsize=10)
    save_fig(fig, "fig07_model_comparison")
    plt.close(fig)


def fig08_spatial_cv_comparison(block_cv_table):
    fold_cols_spatial = [c for c in block_cv_table.columns if c.startswith('Fold')]
    fig, ax = plt.subplots(figsize=(14 / 2.54, 8 / 2.54))
    data, labels, positions = [], [], []
    pos = 0
    for _, row in block_cv_table.iterrows():
        data.append(row[fold_cols_spatial].values.astype(float))
        labels.append(row['Model'])
        positions.append(pos)
        pos += 1
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#90CAF9')
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Spatial block CV R2")
    ax.set_title("Figure 8. Spatial block CV R2 distribution across folds\n"
                  "(compare Mean_R2_spatial vs Mean_R2_standard in Table 5)", fontsize=9)
    save_fig(fig, "fig08_spatial_cv_comparison")
    plt.close(fig)


def fig09_observed_vs_predicted_top4(results, y_test, table4):
    top4 = table4.sort_values('R2', ascending=False).head(4)['Model'].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14 / 2.54, 14 / 2.54))
    for ax, name in zip(axes.flat, top4):
        y_pred = results[name]['y_pred']
        ax.scatter(y_test, y_pred, s=6, alpha=0.3, c=config.PALETTE.get(name, '#1565C0'))
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', lw=1)
        r2 = table4.set_index('Model').loc[name, 'R2']
        rmse = table4.set_index('Model').loc[name, 'RMSE']
        ax.text(0.05, 0.9, f"R2={r2:.3f}\nRMSE={rmse:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Observed (t/ha/yr)"); ax.set_ylabel("Predicted (t/ha/yr)")
    fig.suptitle("Figure 9. Observed vs. predicted -- top 4 models", fontsize=10)
    save_fig(fig, "fig09_observed_vs_predicted_top4")
    plt.close(fig)


def fig10_shap_summary(shap_xgb, shap_stack, X_test, X_test_kernel, feature_names):
    import shap as shap_lib
    fig, axes = plt.subplots(1, 2, figsize=(17 / 2.54, 9 / 2.54))
    plt.sca(axes[0])
    shap_lib.summary_plot(shap_xgb, X_test, feature_names=feature_names, show=False, plot_size=None)
    axes[0].set_title("XGBoost", fontsize=9)
    plt.sca(axes[1])
    shap_lib.summary_plot(shap_stack, X_test_kernel, feature_names=feature_names, show=False, plot_size=None)
    axes[1].set_title("Stacking Ensemble", fontsize=9)
    fig.suptitle("Figure 10. SHAP summary (beeswarm): XGBoost vs Stacking Ensemble", fontsize=10)
    save_fig(fig, "fig10_shap_summary")
    plt.close(fig)


def fig11_shap_dependence(shap_values, X_test, feature_names, top_n=6):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    fig, axes = plt.subplots(2, 3, figsize=(17 / 2.54, 10 / 2.54))
    for ax, i in zip(axes.flat, top_idx):
        ax.scatter(X_test[:, i], shap_values[:, i], s=5, alpha=0.4, c='#2E7D32')
        ax.set_xlabel(feature_names[i], fontsize=8)
        ax.set_ylabel("SHAP value", fontsize=8)
        ax.axhline(0, color='grey', lw=0.6)
    fig.suptitle("Figure 11. SHAP dependence plots -- top 6 features", fontsize=10)
    save_fig(fig, "fig11_shap_dependence")
    plt.close(fig)


def fig12_spatial_shap_maps(shap_rasters, top_features):
    fig, axes = plt.subplots(1, len(top_features), figsize=(17 / 2.54, 4.5 / 2.54))
    if len(top_features) == 1:
        axes = [axes]
    vmax = max(np.nanmax(np.abs(shap_rasters[f]['grid'])) for f in top_features)
    for ax, f in zip(axes, top_features):
        im = ax.imshow(shap_rasters[f]['grid'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f, fontsize=7.5); ax.axis('off')
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("SHAP value", fontsize=7)
    fig.suptitle("Figure 12. Spatial SHAP attribution maps -- top 5 features", fontsize=10)
    save_fig(fig, "fig12_spatial_shap_maps")
    plt.close(fig)


def fig13_uncertainty_maps(uncertainty_rasters):
    fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 6 / 2.54))
    panels = [('mean', 'Mean prediction (t/ha/yr)', 'viridis'),
              ('width', '90% PI width', 'magma'),
              ('cv', 'Coefficient of variation (%)', 'plasma')]
    for ax, (key, title, cmap) in zip(axes, panels):
        im = ax.imshow(uncertainty_rasters[key], cmap=cmap)
        ax.set_title(title, fontsize=8); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 13. Uncertainty maps (Bootstrap + Conformal)", fontsize=10)
    save_fig(fig, "fig13_uncertainty_maps")
    plt.close(fig)


def fig14_erosion_susceptibility_map_final(class_raster):
    cmap = ListedColormap(config.EROSION_COLORS)
    fig, ax = plt.subplots(figsize=(14 / 2.54, 12 / 2.54))
    im = ax.imshow(class_raster, cmap=cmap, vmin=0.5, vmax=6.5)
    ax.axis('off')
    ax.set_title("Figure 14. Final erosion susceptibility map (best ensemble model)", fontsize=10)
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(config.EROSION_COLORS, config.EROSION_LABELS)]
    ax.legend(handles=patches, loc='lower left', fontsize=7, framealpha=0.9)
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='center',
                arrowprops=dict(facecolor='black', width=2, headwidth=8))
    save_fig(fig, "fig14_erosion_susceptibility_map_final")
    plt.close(fig)


def fig15_erosion_class_statistics(class_stats_df):
    fig, axes = plt.subplots(1, 2, figsize=(14 / 2.54, 7 / 2.54))
    axes[0].pie(class_stats_df['Area_km2'], labels=class_stats_df['Class'],
                colors=config.EROSION_COLORS, autopct='%1.1f%%', textprops={'fontsize': 7})
    axes[0].set_title("A. Area proportion per class", fontsize=9)

    axes[1].bar(class_stats_df['Class'], class_stats_df['Area_km2'], color=config.EROSION_COLORS)
    axes[1].set_ylabel("Area (km2)")
    axes[1].tick_params(axis='x', rotation=30, labelsize=7)
    axes[1].set_title("B. Area per erosion class", fontsize=9)
    fig.suptitle("Figure 15. Erosion class statistics", fontsize=10)
    save_fig(fig, "fig15_erosion_class_statistics")
    plt.close(fig)


def fig16_literature_comparison(lit_table, this_study_r2=None):
    fig, ax = plt.subplots(figsize=(14 / 2.54, 8 / 2.54))
    numeric = pd.to_numeric(lit_table.get('R2', pd.Series(dtype=float)), errors='coerce')
    x = np.arange(len(lit_table))
    ax.bar(x, numeric.fillna(0), color='#90A4AE', label='Literature (where reported)')
    if this_study_r2 is not None:
        ax.axhline(this_study_r2, color='#B71C1C', ls='--', lw=1.5,
                    label=f'This study (R2={this_study_r2:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(lit_table['Reference'], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel("R2"); ax.legend(fontsize=7)
    ax.set_title("Figure 16. Comparison with published Morocco/Maghreb erosion studies", fontsize=9)
    save_fig(fig, "fig16_literature_comparison")
    plt.close(fig)


def run(artifacts):
    """artifacts: big dict assembled in main.py with every figure's inputs."""
    set_plot_style()
    fig01_conceptual_framework()
    fig02_study_area(artifacts['coords'])
    fig03_rusle_factor_maps(artifacts['raster_bands'], artifacts['raster_extent'])
    fig04_feature_analysis(artifacts['df_engineered'], artifacts['features'])
    fig05_vif_feature_selection(artifacts['vif_table'])
    fig06_dl_architecture()
    fig07_model_comparison(artifacts['table4'])
    fig08_spatial_cv_comparison(artifacts['block_cv_table'])
    fig09_observed_vs_predicted_top4(artifacts['all_results'], artifacts['y_test'], artifacts['table4'])
    fig10_shap_summary(artifacts['shap_xgb'], artifacts['shap_stack'], artifacts['X_test'],
                        artifacts['X_test_kernel'], artifacts['features'])
    fig11_shap_dependence(artifacts['shap_xgb'], artifacts['X_test'], artifacts['features'])
    fig12_spatial_shap_maps(artifacts['shap_rasters'], artifacts['top_shap_features'])
    fig13_uncertainty_maps(artifacts['uncertainty_rasters'])
    fig14_erosion_susceptibility_map_final(artifacts['erosion_class_raster'])
    fig15_erosion_class_statistics(artifacts['class_stats_df'])
    fig16_literature_comparison(artifacts['literature_table'], artifacts['best_r2'])
    print(f"  All 16 figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    pass
