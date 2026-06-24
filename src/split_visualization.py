import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Visualization functions for the train/validation/test split analysis.

Functions
---------
plot_dst_split_distribution(omni_fmt_path, omni_lst_path, ...)
    Three-panel Dst distribution figure motivated by Cristoforetti et al. (2022) [SCI22].
    Panel (a): chronological split, (b): random split, (c): inverse frequency weighting.
"""



def plot_dst_split_distribution(
    omni_fmt_path : str = 'data/input/omni2_6HWKguLfs2.fmt',
    omni_lst_path : str = 'data/input/omni2_6HWKguLfs2.lst',
    year_start    : int = 1990,
    year_end      : int = 2020,
    train_cutoff  : str = '2009-01-01',
    val_cutoff    : str = '2014-07-01',
    save_path     : str = None,
):
    """
    Three-panel Dst distribution figure — motivated by Cristoforetti et al. (2022) [SCI22].

    Demonstrates that chronological splitting creates distributional mismatch
    in the storm tail between train and val/test segments — motivating storm
    sample weighting (w = 1/f_storm) applied throughout this project.

    Parameters
    ----------
    omni_fmt_path : str — path to OMNI format file
    omni_lst_path : str — path to OMNI data file
    year_start    : int — start year for filtering (default 1990)
    year_end      : int — end year for filtering (default 2020)
    train_cutoff  : str — train/val boundary date (default '2009-01-01')
    val_cutoff    : str — val/test boundary date (default '2014-07-01')
    save_path     : str — if provided, save figure to this path
    """

    # ── Data Preparation ─────────────────────────────────────────────────
    col_names = pd.read_fwf(omni_fmt_path, skiprows=4, header=None)[1].tolist()
    df_omni   = pd.read_csv(omni_lst_path, sep=r'\s+', header=None, names=col_names)
    df_omni['datetime'] = pd.to_datetime(
        df_omni['YEAR'].astype(str) + ' ' +
        df_omni['DOY'].astype(str)  + ' ' +
        df_omni['Hour'].astype(str), format='%Y %j %H'
    )

    mask_period = (df_omni['YEAR'] >= year_start) & (df_omni['YEAR'] < year_end)
    dst_subset  = df_omni.loc[mask_period, ['datetime', 'Dst-index, nT', 'YEAR']].dropna()
    dst_vals    = dst_subset['Dst-index, nT'].values

    # ── Split Strategy (a): Chronological ────────────────────────────────
    val_mask  = (dst_subset['datetime'] >= train_cutoff) & (dst_subset['datetime'] < val_cutoff)
    test_mask = dst_subset['datetime'] >= val_cutoff
    splits_a  = {
        'validation': dst_vals[val_mask.values],
        'test'      : dst_vals[test_mask.values],
    }

    # ── Split Strategy (b): Random ────────────────────────────────────────
    rng                  = np.random.default_rng(42)
    indices              = rng.permutation(len(dst_vals))
    itrain, ival, itest  = np.split(indices, [int(0.7*len(dst_vals)), int(0.85*len(dst_vals))])
    splits_b             = {
        'train'     : dst_vals[itrain],
        'validation': dst_vals[ival],
        'test'      : dst_vals[itest],
    }

    # ── Weighting Logic (c): Inverse Frequency ────────────────────────────
    train_full    = df_omni[df_omni['datetime'] < train_cutoff]['Dst-index, nT'].dropna()
    bins_c        = np.arange(-600, 60, 10)
    counts, edges = np.histogram(train_full, bins=bins_c)
    bin_indices   = np.clip(np.digitize(train_full, edges) - 1, 0, len(counts) - 1)
    weights       = 1.0 / np.maximum(counts[bin_indices], 1)
    weights      /= weights.mean()

    # ── Plotting ──────────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    colors    = {
        'train'     : '#2166ac',
        'validation': '#d6604d',
        'test'      : '#4dac26',
        'weighted'  : '#ef8a62',
    }

    def apply_styling(ax, title):
        ax.set_yscale('log')
        ax.set_xlabel('$D_{st}$ [nT]')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)

    bins_ab = np.arange(-250, 60, 10)

    # Panel (a)
    for label, data in splits_a.items():
        axes[0].hist(data, bins=bins_ab, density=True, histtype='step',
                     label=label, color=colors[label], lw=1.6)
    apply_styling(axes[0], f'(a) Chronological split\n'
                           f'Val {train_cutoff[:4]}–{val_cutoff[:4]}, '
                           f'Test {val_cutoff[:4]}–{year_end}')
    axes[0].set_ylabel('Density')

    # Panel (b)
    for label, data in splits_b.items():
        axes[1].hist(data, bins=bins_ab, density=True, histtype='step',
                     label=label, color=colors[label], lw=1.6)
    apply_styling(axes[1], '(b) Random split\n70% / 15% / 15%')

    # Panel (c)
    axes[2].hist(train_full, bins=bins_c, density=True, histtype='step',
                 label='nominal', color=colors['train'], lw=1.6)
    axes[2].hist(train_full, bins=bins_c, weights=weights, density=True,
                 histtype='step', label='re-weighted', color=colors['weighted'], lw=1.6)
    apply_styling(axes[2], '(c) Train set weighting\nInverse frequency flattening')

    for ax in axes:
        ax.legend(frameon=False)

    plt.suptitle(
        '$D_{st}$ Data Distribution and Split Strategies (Collado-Villaverde et al., 2022)',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()