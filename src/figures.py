# src/figures.py
"""
Figure-generation functions for the project's illustrative and diagram figures.

Extracted from images_generator.ipynb, where each figure was a standalone script cell —
no functions, no explicit save_path parameter (every cell hardcoded a '../images/...' path
assuming the notebook runs from a `notebooks/` subdirectory), and at least one latent bug
(the Burton model figure referenced `ax`, a variable left over from an earlier cell in the
same kernel session, not defined within its own cell — a NameError if run standalone, or a
silent wrong-axes bug if some other `ax` happens to still be in scope).

This is very likely the source of the "sometimes the link breaks" symptom: if
images_generator.ipynb and the notebook embedding the image (e.g.
cosmic_ray_storm_prediction_EDA.ipynb, via `<img src="images/...">`) are run from different
working directories, a hardcoded '../images/...' path resolves correctly from only one of
them. Every function below instead takes `save_path` as an explicit parameter with no
default — exactly the pattern already used in `split_visualization.py`'s
`plot_dst_split_distribution()` — so the caller decides the path, and it can be run from
any working directory as long as the caller passes the correct one.

Functions
---------
plot_sun_l1_earth_chain(save_path=None)
    Figure 1 — schematic of the Sun-CME-L1-Earth causal chain (synthetic/illustrative).

plot_forbush_decrease_dst(save_path=None)
    Figure 2 — synthetic Forbush Decrease precursor and Dst storm response profile.

plot_burton_model(save_path=None)
    Figure 3 — Burton et al. (1975) [BUR75] injection function and storm-decay ODE.

plot_synthetic_solar_cycle(save_path=None)
    Figure 4 (synthetic) — illustrative solar cycle / storm frequency relationship.

plot_solar_cycle_from_omni(omni_lst_path, save_path=None)
    Figure 4 (real data) — annual sunspot number and storm counts from NASA OMNI [PAP20].

plot_feature_selection_pipeline(save_path=None)
    Feature Selection Pipeline architecture diagram (LassoSelector + ExtraTreesSelector,
    majority-vote consolidation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle


def _apply_style():
    """Shared rcParams for all figures in this module (matches images_generator.ipynb Cell 1)."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_sun_l1_earth_chain(save_path: str = None):
    """
    Figure 1 — schematic of the Sun -> CME -> L1 -> Earth causal chain.

    Synthetic/illustrative only — does not plot measured data.

    Parameters
    ----------
    save_path : str or None — if provided, save the figure to this exact path
                (no directory prefix is assumed; pass the full relative or
                absolute path from your own working directory)
    """
    _apply_style()
    fig1, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 2.5)
    ax.axis('off')
    ax.set_facecolor('#0a0a1a')
    fig1.patch.set_facecolor('#0a0a1a')

    sun = Circle((0.8, 0.5), 0.6, color='#FDB813', zorder=5)
    ax.add_patch(sun)
    ax.text(0.8, -0.4,  'Sun',                  color='white',   ha='center', fontsize=12, fontweight='bold')
    ax.text(0.8, -0.75, '(solar cycle\n~11 years)', color='#aaaaaa', ha='center', fontsize=8.5)

    cme = Ellipse((3.5, 0.5), 1.2, 0.7, color='#FF6B35', alpha=0.7, zorder=4)
    ax.add_patch(cme)
    ax.text(3.5, -0.4,  'CME + shock',    color='white',   ha='center', fontsize=10)
    ax.text(3.5, -0.72, '300–2000 km/s', color='#aaaaaa', ha='center', fontsize=8.5)

    l1 = Circle((6.2, 0.5), 0.18, color='#00BFFF', zorder=5)
    ax.add_patch(l1)
    ax.text(6.2, -0.4,  'L1',                          color='white',   ha='center', fontsize=12, fontweight='bold')
    ax.text(6.2, -0.72, '1.5×10⁶ km\nfrom Earth',     color='#aaaaaa', ha='center', fontsize=8.5)
    ax.text(6.2,  1.05, 'measures: $V_{sw}$, $B_z$, $n_p$, $T$', color='#00BFFF', ha='center', fontsize=9)

    earth = Circle((8.8, 0.5), 0.45, color='#1a6b3a', zorder=5)
    ax.add_patch(earth)
    mag = Ellipse((8.5, 0.5), 2.2, 1.4, color='#4169E1', alpha=0.15, zorder=3)
    ax.add_patch(mag)
    ax.text(8.8, -0.4,  'Earth',                  color='white',   ha='center', fontsize=12, fontweight='bold')
    ax.text(8.8, -0.72, 'magnetosphere\nring current', color='#aaaaaa', ha='center', fontsize=8.5)

    for xy_start, xy_end in [((1.4, 0.5), (2.85, 0.5)), ((4.15, 0.5), (5.95, 0.5)), ((6.42, 0.5), (8.25, 0.5))]:
        ax.annotate('', xy=xy_end, xytext=xy_start,
                    arrowprops=dict(arrowstyle='->', color='#FF6B35', lw=2, mutation_scale=18))

    ax.annotate('', xy=(3.5, 1.7), xytext=(3.5, 2.3),
                arrowprops=dict(arrowstyle='->', color='#aaaaff', lw=1.5, mutation_scale=14))
    ax.text(3.5, 2.35, 'galactic cosmic rays\n(Forbush Decrease ↓)',
            color='#aaaaff', ha='center', fontsize=8.5)

    ax.annotate('', xy=(8.25, 1.55), xytext=(6.42, 1.55),
                arrowprops=dict(arrowstyle='<->', color='#00BFFF', lw=1.5, mutation_scale=12))
    ax.text(7.33, 1.72, '15–60 min lead time', color='#00BFFF', ha='center', fontsize=8.5)

    ax.set_title('Figure 1 — The Sun–L1–Earth chain: physical stages of a geomagnetic storm',
                 color='white', pad=12, fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_forbush_decrease_dst(save_path: str = None):
    """
    Figure 2 — synthetic Forbush Decrease precursor and Dst storm response.

    Synthetic/illustrative only — does not plot measured data.

    Parameters
    ----------
    save_path : str or None — if provided, save the figure to this exact path
    """
    _apply_style()
    np.random.seed(42)
    t = np.linspace(0, 120, 1200)
    baseline  = 5200 + 30 * np.sin(2 * np.pi * t / 130)
    fd_profile = np.zeros(len(t))
    fd_profile[(t >= 38) & (t < 52)] = -220 * ((t[(t >= 38) & (t < 52)] - 38) / 14) ** 1.3
    fd_profile[(t >= 52) & (t < 58)] = -220
    fd_profile[(t >= 58) & (t < 90)] = -220 * np.exp(-0.055 * (t[(t >= 58) & (t < 90)] - 58))
    counts = baseline + fd_profile + np.random.normal(0, 18, len(t))

    dst_base = np.zeros(len(t))
    dst_base[(t >= 48) & (t < 65)] = -150 * ((t[(t >= 48) & (t < 65)] - 48) / 17) ** 1.5
    dst_base[(t >= 65) & (t < 110)] = -150 * np.exp(-0.045 * (t[(t >= 65) & (t < 110)] - 65))
    dst = dst_base + np.random.normal(0, 4, len(t))

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, counts, color='#2166ac', lw=1.2, label='Neutron count rate (synthetic)')
    ax1.axvline(38, color='orange', lw=1.5, ls='--', label='CME shock at L1')
    ax1.axvline(48, color='red',    lw=1.5, ls='--', label='Storm onset at Earth')
    ax1.set_ylabel('Count rate [counts / min]')
    ax1.set_title('Figure 2 — Synthetic Forbush Decrease and storm response', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.annotate('Forbush\nDecrease\n~4–8%', xy=(52, baseline.mean() - 200),
                 xytext=(62, baseline.mean() - 350),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=9, color='darkred')

    ax2.plot(t, dst, color='#d6604d', lw=1.4, label='$D_{st}$ index (synthetic)')
    ax2.axhline(-50,  color='orange', lw=1, ls=':', alpha=0.7, label='Moderate storm (−50 nT)')
    ax2.axhline(-100, color='red',    lw=1, ls=':', alpha=0.7, label='Strong storm (−100 nT)')
    ax2.axvline(38, color='orange', lw=1.5, ls='--')
    ax2.axvline(48, color='red',    lw=1.5, ls='--')
    ax2.fill_between(t, dst, 0, where=(dst < -50), alpha=0.12, color='red')
    ax2.set_ylabel('$D_{st}$ [nT]')
    ax2.set_xlabel('Time [hours from CME eruption]')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.annotate('', xy=(48, -165), xytext=(38, -165),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=1.5, mutation_scale=12))
    ax2.text(43, -172, '~10 h\nlead time', ha='center', fontsize=8.5, color='darkgreen')

    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_burton_model(save_path: str = None):
    """
    Figure 3 — Burton et al. (1975) [BUR75] injection function Q(Ey) and storm-decay ODE.

    Fix applied vs. the original notebook cell: `ax.set_aspect('equal')` was called on
    `ax`, a variable left over from an earlier, unrelated cell in the same kernel session
    (this cell itself only defines `axes[0]`, `ax_r`, `ax_r2` — never a bare `ax`). That
    line is removed here since it targeted the wrong axes object (or would raise a
    NameError entirely when this cell is run on its own).

    Parameters
    ----------
    save_path : str or None — if provided, save the figure to this exact path
    """
    _apply_style()
    Ey = np.linspace(-1, 6, 500)
    Q  = np.where(Ey > 0.5, -4.4 * (Ey - 0.5), 0.0)

    t2  = np.linspace(0, 48, 4800)
    dt  = t2[1] - t2[0]
    tau = 7.5
    Ey_t = np.zeros(len(t2))
    Ey_t[(t2 >= 5)  & (t2 < 8)]  = np.linspace(0, 4.5, ((t2 >= 5)  & (t2 < 8)).sum())
    Ey_t[(t2 >= 8)  & (t2 < 20)] = 4.5
    Ey_t[(t2 >= 20) & (t2 < 24)] = np.linspace(4.5, 0.3, ((t2 >= 20) & (t2 < 24)).sum())
    Ey_t[t2 >= 24]  = 0.3
    Q_t   = np.where(Ey_t > 0.5, -4.4 * (Ey_t - 0.5), 0.0)
    Dst2  = np.zeros(len(t2))
    for i in range(1, len(t2)):
        Dst2[i] = Dst2[i-1] + dt * (Q_t[i-1] - Dst2[i-1] / tau)

    fig3, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(Ey, Q, color='#d6604d', lw=2.5)
    axes[0].axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
    axes[0].axvline(0.5, color='gray', lw=1, ls=':', label='Threshold $E_y = 0.5$ mV/m')
    axes[0].fill_between(Ey, Q, 0, where=(Q < 0), alpha=0.12, color='red')
    axes[0].set_xlabel('$E_y = -V_{sw} \\cdot B_z \\cdot 10^{-3}$ [mV/m]')
    axes[0].set_ylabel('$Q(E_y)$ [nT/h]')
    axes[0].set_title('Injection function $Q(E_y)$')
    axes[0].legend(fontsize=9)
    axes[0].text(3.5, -12, '$Q = -4.4\\,(E_y - 0.5)$', fontsize=10, color='darkred')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    ax_r  = axes[1]
    ax_r2 = ax_r.twinx()
    ax_r2.plot(t2, Ey_t, color='#2166ac', lw=1.5, ls='--', alpha=0.7, label='$E_y(t)$')
    ax_r2.set_ylabel('$E_y$ [mV/m]', color='#2166ac')
    ax_r2.tick_params(axis='y', labelcolor='#2166ac')
    ax_r2.spines['top'].set_visible(False)
    ax_r.plot(t2, Dst2, color='#d6604d', lw=2.2, label='$D_{st}(t)$ — Burton model')
    ax_r.axhline(-50,  color='orange', lw=1, ls=':', alpha=0.6)
    ax_r.axhline(-100, color='red',    lw=1, ls=':', alpha=0.6)
    ax_r.set_xlabel('Time [hours]')
    ax_r.set_ylabel('$D_{st}$ [nT]', color='#d6604d')
    ax_r.set_title('Storm evolution: Burton (1975) ODE [BUR75]')
    ax_r.tick_params(axis='y', labelcolor='#d6604d')
    ax_r.annotate('decay $\\tau \\approx 7.5$ h', xy=(30, -80), xytext=(35, -55),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=9)
    lines1, labels1 = ax_r.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower right')
    ax_r.spines['top'].set_visible(False)
    # NOTE: the original cell called ax.set_aspect('equal') here, referencing a leftover
    # `ax` from a previous cell — removed, see docstring.
    plt.suptitle('Figure 3 — Burton (1975) physical model [BUR75]', fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_synthetic_solar_cycle(save_path: str = None):
    """
    Figure 4 (synthetic) — illustrative solar cycle / storm frequency relationship.

    Synthetic/illustrative only — does not plot measured data. For the real-data
    equivalent, see plot_solar_cycle_from_omni().

    Parameters
    ----------
    save_path : str or None — if provided, save the figure to this exact path
    """
    _apply_style()
    np.random.seed(7)
    years = np.linspace(1981, 2023, 1000)

    def solar_cycle(t):
        result = np.zeros_like(t)
        for tmin, period in [(1986, 10.0), (1996, 10.2), (2008, 9.9), (2019, 11.0)]:
            result += np.clip(np.sin(2 * np.pi * (t - tmin) / period) * 160 + 80, 0, None)
        return result / 4

    ssn = solar_cycle(years) + np.random.normal(0, 8, len(years))
    ssn = np.clip(ssn, 0, None)
    year_bins   = np.arange(1981, 2024)
    ssn_annual  = np.array([solar_cycle(np.array([y + 0.5]))[0] for y in year_bins])
    storm_counts = np.clip(np.round(ssn_annual / 30 + np.random.normal(0, 0.8, len(year_bins))).astype(int), 0, None)

    fig4, (ax_ssn, ax_st) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax_ssn.fill_between(years, ssn, color='#FDB813', alpha=0.6)
    ax_ssn.plot(years, ssn, color='#b8860b', lw=1)
    ax_ssn.set_ylabel('Sunspot number\n(synthetic)')
    ax_ssn.set_title('Figure 4 (synthetic) — Solar cycle and storm frequency', fontsize=12)
    ax_ssn.spines['top'].set_visible(False)
    ax_ssn.spines['right'].set_visible(False)

    ax_st.bar(year_bins, storm_counts, color='#d6604d', width=0.8)
    ax_st.set_ylabel('Storms/year\n($D_{st}<-100$ nT, synthetic)')
    ax_st.set_xlabel('Year')
    ax_st.spines['top'].set_visible(False)
    ax_st.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_solar_cycle_from_omni(omni_lst_path: str, save_path: str = None):
    """
    Figure 4 (real data) — annual sunspot number and storm counts from NASA OMNI [PAP20].

    Column indices follow the OMNI format file [KIN05]: col 0 = YEAR, col 41 = R (Sunspot
    Number), col 42 = Dst index [nT]. Fill values (999 for R, 99999 for Dst) are converted
    to NaN per OMNI convention [KIN05].

    Parameters
    ----------
    omni_lst_path : str — path to the OMNI .lst data file
    save_path     : str or None — if provided, save the figure to this exact path

    Returns
    -------
    pandas.DataFrame — the annual aggregation (year, R_mean, storm counts), for reuse
    """
    _apply_style()
    col_names = [str(i) for i in range(57)]
    df = pd.read_csv(omni_lst_path, sep=r'\s+', header=None, names=col_names)

    df['year'] = df['0'].astype(int)
    df['R']    = pd.to_numeric(df['41'], errors='coerce')
    df['Dst']  = pd.to_numeric(df['42'], errors='coerce')

    df.loc[df['R']   == 999,   'R']   = np.nan
    df.loc[df['Dst'] == 99999, 'Dst'] = np.nan

    annual = df.groupby('year').agg(
        R_mean     = ('R',   'mean'),
        storms_str = ('Dst', lambda x: (x < -100).sum()),
    ).reset_index()

    fig, (ax_ssn, ax_st) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax_ssn.fill_between(annual['year'], annual['R_mean'], color='#FDB813', alpha=0.6)
    ax_ssn.plot(annual['year'], annual['R_mean'], color='#b8860b', lw=1)
    ax_ssn.set_ylabel('Sunspot number\n(annual mean)')
    ax_ssn.set_title('Figure 4 — Solar cycle non-stationarity (NASA OMNI [PAP20])', fontsize=12)
    ax_ssn.spines['top'].set_visible(False)
    ax_ssn.spines['right'].set_visible(False)

    ax_st.bar(annual['year'], annual['storms_str'], color='#d6604d', width=0.8)
    ax_st.set_ylabel('Storms/year\n($D_{st}<-100$ nT)')
    ax_st.set_xlabel('Year')
    ax_st.spines['top'].set_visible(False)
    ax_st.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

    return annual


def plot_feature_selection_pipeline(save_path: str = None):
    """
    Feature Selection Pipeline architecture diagram — LassoSelector + ExtraTreesSelector,
    majority-vote consolidation, per-horizon survivor counts, and the physics-motivated
    force-inclusion of d_neutron [KIS25].

    The specific counts shown (23/33 strict-vote survivors, 24/33 final selection) reflect
    the feature_selection.ipynb run this diagram documents — if that pipeline is re-run and
    produces different counts, update the labels below accordingly rather than assuming
    they still match.

    Parameters
    ----------
    save_path : str or None — if provided, save the figure to this exact path
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def box(ax, x, y, w, h, text, color='#2166ac', textcolor='white', fontsize=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, zorder=3,
                                    facecolor=color, edgecolor='white', linewidth=1.5))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                color=textcolor, fontsize=fontsize, fontweight='bold',
                zorder=4, multialignment='center')

    def arrow(ax, x1, y1, x2, y2, color='#444444'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                    mutation_scale=16))

    box(ax, 0.3, 2.3, 2.4, 1.4,
        'X_train\n121,185 rows\n33 features',
        color='#4d4d4d', fontsize=9)

    arrow(ax, 2.7, 3.7, 3.8, 4.5)
    arrow(ax, 2.7, 2.3, 3.8, 1.5)

    box(ax, 3.8, 3.8, 3.2, 1.4,
        'LassoSelector\nLinear · L1\nTimeSeriesSplit CV\nstorm-weighted',
        color='#2166ac', fontsize=8.5)

    box(ax, 3.8, 0.8, 3.2, 1.4,
        'ExtraTreesSelector\nNon-linear · ensemble\nthreshold_q=0.50\nstorm-weighted',
        color='#1a9850', fontsize=8.5)

    ax.text(5.4, 5.55, '× 5 horizons  (h = 1, 3, 7, 12, 21h)',
            ha='center', va='center', fontsize=9, color='#555555', style='italic')

    arrow(ax, 7.0, 4.5, 8.1, 4.5)
    arrow(ax, 7.0, 1.5, 8.1, 1.5)

    box(ax, 8.1, 3.8, 2.2, 1.4,
        'survivors_lasso\n17–22 / 33\nper horizon',
        color='#4393c3', fontsize=8.5)

    box(ax, 8.1, 0.8, 2.2, 1.4,
        'survivors_et\n17 / 33\nper horizon',
        color='#41ab5d', fontsize=8.5)

    arrow(ax, 10.3, 4.5, 11.0, 3.7)
    arrow(ax, 10.3, 1.5, 11.0, 2.3)

    box(ax, 11.0, 2.3, 2.2, 1.4,
        'Majority Vote\nmin_votes=2\nunion across\nhorizons',
        color='#7b2d8b', fontsize=8.5)

    arrow(ax, 13.2, 3.0, 13.5, 3.0)

    box(ax, 13.5, 2.3, 2.2, 1.4,
        'SELECTED\nFEATURES\n24 / 33',
        color='#d6604d', fontsize=8.5)

    ax.text(14.6, 1.6, '+ d_neutron\nforce-included [KIS25]',
            ha='center', va='center', fontsize=8, color='#d6604d', style='italic')
    ax.annotate('', xy=(14.6, 2.3), xytext=(14.6, 2.0),
                arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.2))

    ax.set_title(
        'Feature Selection Pipeline Architecture\n'
        'LassoSelector + ExtraTreesSelector · storm-weighted · 5 horizons',
        fontsize=12, fontweight='bold', pad=12
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()
