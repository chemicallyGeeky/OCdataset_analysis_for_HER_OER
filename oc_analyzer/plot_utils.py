import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_main_panels(ae_limits=(-2, 2), eta_limits=(2, 0), xlabel=r'${\Delta G}_H$'):

    fig = plt.figure(figsize=(12, 8))
    # Set up the GridSpec layout: 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1],
                           height_ratios=[1, 4], wspace=0.05, hspace=0.05)
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
    ax_main_left.set_ylabel(r'$\eta_{TD}$ (V)', fontsize=20)
    ax_top.set_title(xlabel + r" distribution", fontsize=20)
    ax_right.set_title(r'$\eta_{TD}$ distribution', rotation=-90, x=1.2, y=0.2, fontsize=20)

    return fig, ax_main_left, ax_top, ax_right


def add_shadded_regions(ax_main_left, ax_top, ax_right, uncertainty=0.3):

    # Add shaded regions for uncertainty
    ax_main_left.axhspan(uncertainty, 0, color='gray', alpha=0.2,
                         label="Ideal within uncertainty", zorder=0)
    ax_main_left.axvspan(-uncertainty, uncertainty, color='gray', alpha=0.2, zorder=0)
    ax_top.axvspan(-uncertainty, uncertainty, color='gray', alpha=0.2, zorder=0)
    ax_right.axhspan(uncertainty, 0, color='gray', alpha=0.2, zorder=0)


def plot_main_panel(ax_main_left, ax_top, ax_right, her_data,
                    xlabel, lit=None, special_samples=None):

    ax_main_left.scatter(her_data[xlabel], her_data['eta'],
                         color='k', s=5, alpha=0.1, label='OC20 DFT predictions')

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


def plot_distributions(ax_top, ax_right, her_data, xlabel, uncertainty=0):

    # Shaded area
    counts, bins = np.histogram(her_data[xlabel], bins=150)         # top plot
    bins = (bins[1:] + bins[:-1]) / 2
    w = bins[1] - bins[0]
    ax_top.bar(bins[(bins > -uncertainty) & (bins < uncertainty)],
               counts[(bins > -uncertainty) & (bins < uncertainty)],
               width=w * 0.8, color='cornflowerblue', alpha=1)    # top plot
    ax_top.bar(bins[(bins < -uncertainty) | (bins > uncertainty)],
               counts[(bins < -uncertainty) | (bins > uncertainty)],
               width=w * 0.8, color='cornflowerblue', alpha=uncertainty)    # top plot


    counts, bins = np.histogram(her_data['eta'], bins=150)    # right plot
    bins = (bins[1:] + bins[:-1]) / 2
    w = bins[1] - bins[0]
    ax_right.barh(bins[bins > uncertainty], counts[bins > uncertainty], height=w * 0.8,
                  color='cornflowerblue', alpha=0.3)    # right plot
    ax_right.barh(bins[bins < uncertainty], counts[bins < uncertainty], height=w * 0.8,
                  color='cornflowerblue', alpha=1)    # right plot
