# src/selection_visualization.py
"""
Visualization functions for the feature selection pipeline.

Functions
---------
plot_lasso_panel(ax, coef, title)
    Bar chart of LASSO coefficients. d_neutron highlighted in red.

plot_et_panel(ax, importances, title)
    Bar chart of ExtraTrees importances with median threshold line.

plot_lasso_vs_et(coef, importances, horizon, save_path)
    Two-panel figure: LASSO coefficients vs ExtraTrees importances.

plot_et_heatmap(imp_matrix, save_path)
    Heatmap of ExtraTrees importances across all horizons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_lasso_panel(
    ax         : plt.Axes,
    coef       : pd.Series,
    title      : str = 'LASSO Coefficients (storm-weighted)',
    highlight  : str = 'd_neutron',
):
    """
    Bar chart of LASSO coefficients sorted by absolute value.
    Retained features in green, zeroed in grey, highlight feature in red.

    Parameters
    ----------
    ax        : matplotlib Axes
    coef      : Series — feature coefficients (index = feature names)
    title     : str — plot title
    highlight : str — feature name to highlight in red (default 'd_neutron')
    """
    coef   = coef.sort_values(key=abs, ascending=True)
    colors = [
        '#e74c3c' if f == highlight else
        '#2ecc71' if abs(v) > 1e-5 else '#bdc3c7'
        for f, v in coef.items()
    ]

    ax.barh(coef.index, coef.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(0,     color='black', linewidth=0.8)
    ax.axvline( 1e-5, color='gray',  linewidth=0.8, linestyle='--')
    ax.axvline(-1e-5, color='gray',  linewidth=0.8, linestyle='--', label='threshold 1e-5')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Coefficient')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(handles=[
        mpatches.Patch(color='#2ecc71', label='Retained'),
        mpatches.Patch(color='#bdc3c7', label='Zeroed'),
        mpatches.Patch(color='#e74c3c', label=highlight),
    ], fontsize=9)


def plot_et_panel(
    ax         : plt.Axes,
    importances: pd.Series,
    title      : str = 'ExtraTrees Importance (storm-weighted)',
    highlight  : str = 'd_neutron',
):
    """
    Bar chart of ExtraTrees importances sorted ascending.
    Features above median in green, below in grey, highlight in red.
    Median threshold shown as dashed vertical line.

    Parameters
    ----------
    ax          : matplotlib Axes
    importances : Series — feature importances (index = feature names)
    title       : str — plot title
    highlight   : str — feature name to highlight in red (default 'd_neutron')
    """
    imp        = importances.sort_values(ascending=True)
    median_thr = imp.quantile(0.50)
    colors     = [
        '#e74c3c' if f == highlight else
        '#2ecc71' if v >= median_thr else '#bdc3c7'
        for f, v in imp.items()
    ]

    ax.barh(imp.index, imp.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(median_thr, color='navy', linewidth=1.2, linestyle='--',
               label=f'Median threshold ({median_thr:.4f})')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Importance')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(handles=[
        mpatches.Patch(color='#2ecc71', label='Retained (≥ median)'),
        mpatches.Patch(color='#bdc3c7', label='Below median'),
        mpatches.Patch(color='#e74c3c', label=highlight),
    ], fontsize=9)


def plot_lasso_vs_et(
    coef        : pd.Series,
    importances : pd.Series,
    horizon     : int  = 7,
    highlight   : str  = 'd_neutron',
    save_path   : str  = None,
):
    """
    Two-panel figure: LASSO coefficients vs ExtraTrees importances
    at a given forecast horizon.

    Parameters
    ----------
    coef        : Series — LASSO coefficients from LassoSelector.coef_
    importances : Series — ExtraTrees importances from ExtraTreesSelector.importances_
    horizon     : int — forecast horizon in hours (used in title)
    highlight   : str — feature to highlight in red (default 'd_neutron')
    save_path   : str — if provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    plot_lasso_panel(
        axes[0], coef,
        title     = f'LASSO Coefficients — h={horizon}h (storm-weighted)',
        highlight = highlight,
    )
    plot_et_panel(
        axes[1], importances,
        title     = f'ExtraTrees Importance — h={horizon}h (storm-weighted)',
        highlight = highlight,
    )

    plt.suptitle(
        f'Feature Selection — LASSO vs ExtraTrees at h={horizon}h',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_et_heatmap(
    imp_matrix : dict,
    highlight  : str = 'd_neutron',
    save_path  : str = None,
):
    """
    Heatmap of ExtraTrees feature importances across all forecast horizons.
    Features sorted by mean importance descending. Highlight feature outlined in red.

    Parameters
    ----------
    imp_matrix : dict — {label: pd.Series of importances} e.g. {'h=1h': Series, ...}
    highlight  : str — feature to outline in red (default 'd_neutron')
    save_path  : str — if provided, save figure to this path
    """
    imp_df = pd.DataFrame(imp_matrix)
    imp_df = imp_df.loc[imp_df.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 11))

    sns.heatmap(
        imp_df,
        ax         = ax,
        cmap       = 'YlOrRd',
        linewidths = 0.3,
        linecolor  = 'white',
        annot      = True,
        fmt        = '.4f',
        annot_kws  = {'size': 7},
        cbar_kws   = {'label': 'Feature Importance'},
    )

    if highlight in imp_df.index:
        idx = list(imp_df.index).index(highlight)
        ax.add_patch(plt.Rectangle(
            (0, idx), len(imp_df.columns), 1,
            fill=False, edgecolor='#e74c3c', lw=2.5, clip_on=False,
        ))

    ax.set_title(
        f'ExtraTrees Feature Importance — All Horizons\n({highlight} outlined in red)',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()