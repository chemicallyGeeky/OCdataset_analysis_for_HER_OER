import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def create_main_panels(ae_limits=(-2, 2), eta_limits=(2, 0), xlabel=r'${\Delta G}_H$'):

    fig = plt.figure(figsize=(12, 8))
    # Set up the GridSpec layout: 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1],
                           height_ratios=[2, 7], wspace=0.05, hspace=0.05)
    # Main plot: bottom-left
    ax_main_left = fig.add_subplot(gs[1, 0])
    ax_main_left.set_zorder(2)
    ax_main_left.patch.set_visible(False)

    # Top plot: top-left, shares x with main
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main_left)
    # Right plot: bottom-right, shares y with main
    ax_right = fig.add_subplot(gs[1, 1])
    # Axis
    # top
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_top.get_yticklabels(), visible=False)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_top.tick_params(axis='y', which='both', left=False, right=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_right.tick_params(axis='y', which='both', left=False, right=False)
    ax_main_left.tick_params(labelsize=15)

    # Limits
    ax_main_left.set_xlim(ae_limits)
    ax_main_left.set_ylim(eta_limits)
    ax_right.set_ylim(eta_limits)

    # Titles
    ax_main_left.set_xlabel(r'Predicted ' + xlabel + r' (eV)', fontsize=20)
    ax_main_left.set_ylabel(r'$\eta$ (V)', fontsize=20)
    ax_top.set_title(xlabel + r" distribution", fontsize=20)
    ax_right.set_title(r'$\eta$ distribution', rotation=-90, x=1.2, y=0.2, fontsize=20)

    return fig, ax_main_left, ax_top, ax_right


def add_shadded_regions(ax_main_left, ax_top, ax_right, uncertainty=0.3):

    # Add shaded regions for uncertainty
    ax_main_left.axhspan(uncertainty, 0, color='gray', alpha=0.2,
                         label="Ideal within uncertainty", zorder=0)
    ax_main_left.axvspan(-uncertainty, uncertainty, color='gray', alpha=0.2, zorder=0)
    ax_top.axvspan(-uncertainty, uncertainty, color='gray', alpha=0.2, zorder=0)
    ax_right.axhspan(uncertainty, 0, color='gray', alpha=0.2, zorder=0)

def add_best_value(ax_main_left, ax_top, ax_right, best_val=0.4, color='magenta'):

    # Add shaded regions for uncertainty
    ax_main_left.axhline(best_val, 0, color=color, alpha=0.5,
                         label="Best known calatyst", zorder=0)
    ax_main_left.axvline(best_val, 0, color=color, alpha=0.5,
                         zorder=0)
    ax_top.axvline(best_val, color=color, alpha=0.5, zorder=0)
    ax_right.axhline(best_val, color=color, alpha=0.5, zorder=0)

def plot_main_panel(ax_main_left, ax_top, ax_right, her_data,
                    xlabel, lit=None, special_samples=None, s=5, alpha=0.1, color="k", label='OC20 DFT predictions'):

    ax_main_left.scatter(her_data[xlabel], her_data['eta'],
                         color=color, s=s, alpha=alpha, label=label)

    if lit is not None:
        ax_main_left.scatter(lit["database_value"], lit["experimental_value"],
                             color='orange', marker='v', s=30, label='Experimental Data')
        for i, lit_val in lit.iterrows():
            ax_main_left.annotate(i, (lit_val["database_value"], lit_val["HClO4"]),
                                  xytext=(0, 4), textcoords='offset points',
                                  fontsize=8, annotation_clip=False,
                                  horizontalalignment="center",
                                  verticalalignment='bottom', color='orange')

    if special_samples is not None:

        for sample, props in special_samples.items():
            best_idx = her_data[her_data['bulk_symbols'] == sample]['eta'].idxmin()
            x = her_data.loc[best_idx][xlabel]
            y = her_data.loc[best_idx]['eta']
            alignment = 'right' if x < 0 else 'left'
            ax_main_left.scatter(x, y, color=props["color"], s=30, marker='.', alpha=1, zorder=3)
            text_location = np.array([x + np.sign(x) * 0.3, y - props["manual_adjustment"]])
            ax_main_left.annotate(props["description"], (x, y),
                                  xytext=text_location, textcoords='data',
                                  arrowprops=dict(arrowstyle='->'),
                                  fontsize=12, annotation_clip=True,
                                  horizontalalignment=alignment, verticalalignment='center')

    ax_main_left.legend(loc='lower center', fontsize=15)


def plot_distributions(ax_top, ax_right, her_data, xlabel, uncertainty=0, bins=150, color='cornflowerblue'):

    # Shaded area
    counts, bins = np.histogram(her_data[xlabel], bins=bins)         # top plot
    cbins = (bins[1:] + bins[:-1]) / 2
    w = cbins[1] - cbins[0]
    ax_top.bar(cbins[(cbins > -uncertainty) & (cbins < uncertainty)],
               counts[(cbins > -uncertainty) & (cbins < uncertainty)],
               width=w * 0.8, color=color, alpha=1)    # top plot
    ax_top.bar(cbins[(cbins < -uncertainty) | (cbins > uncertainty)],
               counts[(cbins < -uncertainty) | (cbins > uncertainty)],
               width=w * 0.8, color=color, alpha=0.3)    # top plot


    # Reoriganize the bins because this spans only half the range
    half_bins = np.sort(np.concat([(bins[1:] + bins[:-1]) / 2, bins]))
    half_bins = half_bins - half_bins.min()
    half_bins = half_bins[half_bins >= 0]

    counts, half_bins = np.histogram(her_data['eta'], bins=half_bins)    # right plot
    half_cbins = (half_bins[1:] + half_bins[:-1]) / 2
    w = half_cbins[1] - half_cbins[0]
    ax_right.barh(half_cbins[half_cbins > uncertainty], counts[half_cbins > uncertainty],
                  height=w * 0.8, color=color, alpha=0.3)    # right plot
    ax_right.barh(half_cbins[half_cbins < uncertainty], counts[half_cbins < uncertainty],
                  height=w * 0.8, color=color, alpha=1)    # right plot

    return bins

def make_bar_plot(data):
    sns.set(font_scale=3, rc={'font.weight': 'bold'})
    sns.set_style('ticks')
    f, ax = plt.subplots(figsize=(10, 12))
    sns.histplot(y=data['ads_symbols'], color='green')
    ax.set_xlabel('Adsorbate', fontsize=32, fontweight='bold')
    ax.set_ylabel('Count', fontsize=32, fontweight='bold')


def make_violin_plot(data):
    sns.set(font_scale=2, rc={'font.weight': 'bold'})
    sns.set_style('ticks')
    f, ax = plt.subplots(figsize=(10, 10))
    sns.violinplot(data=data.loc[:, ['OH', 'O', 'HO2', 'O2']],
                   x="ads_symbols", y='adsorption_free_energy',
                   palette=sns.color_palette('dark'),
                   fill=False, split=False, inner='quart')
    ax.set_xlabel('Distribution', fontsize=32, fontweight='bold')
    ax.set_ylabel('Adsorption Energy (eV)', fontsize=32, fontweight='bold')

def plot_stability_distribution(stability, stability_filter, uncertainty):

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    counts, bins = np.histogram(stability["decomposition_energy"], bins=150)
    cbins = (bins[1:] + bins[:-1]) / 2
    w = cbins[1] - cbins[0]
    ax.bar(cbins[(cbins < uncertainty)],
           counts[(cbins < uncertainty)],
           width=w * 0.8, color="gray", alpha=1, label=r"Materials with computed $\eta$")
    ax.bar(cbins[(cbins > uncertainty)],
           counts[(cbins > uncertainty)],
           width=w * 0.8, color="gray", alpha=0.3)

    counts, bins = np.histogram(stability.loc[stability_filter, "decomposition_energy"], bins=bins)
    cbins = (bins[1:] + bins[:-1]) / 2
    w = cbins[1] - cbins[0]
    ax.bar(cbins[(cbins < uncertainty)],
           counts[(cbins < uncertainty)],
           width=w * 0.8, color="cornflowerblue", alpha=1, label="Filtered materials")
    ax.bar(cbins[(cbins > uncertainty)],
           counts[(cbins > uncertainty)],
           width=w * 0.8, color="cornflowerblue", alpha=0.3)

    ax.axvspan(stability["decomposition_energy"].min(), uncertainty, color='green', alpha=0.2, zorder=0, lw=0, label="Stable within uncertainty")
    
    ax.set_xlim((stability["decomposition_energy"].min(), 3))
    ax.set_xlabel('Decomposition Energy (eV/atom)', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)
    ax.legend(loc='upper right', fontsize=15)
