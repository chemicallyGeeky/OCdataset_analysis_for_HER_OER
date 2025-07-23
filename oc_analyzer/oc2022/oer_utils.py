import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
plt.rcParams['mathtext.default'] = 'regular'

def gibbs_correction(row, correction_dict):
    """
    Applies Gibbs free energy correction based on the adsorbate symbol.
    Assumes adsorption energy *per adsorbate*.
    """
    return correction_dict.get(row["ads_symbols"], 0) + row["adsorption_energy"]

def scaling(x_vals, y_vals):
    reg = LinearRegression()
    reg.fit(x_vals.reshape(-1, 1), y_vals)
    return reg.intercept_, reg.coef_[0]

def merge_slabs_with_all_adsorbates(df, ads_list, miller=False):

    if miller:
        matching_columns = ['slab_sid', "miller_index"]
    else:
        matching_columns = 'slab_sid'

    merged_df = df[df['ads_symbols'] == ads_list[0]].copy()
    for ads in ads_list[1:]:
        ads_df = df[df['ads_symbols'] == ads]
        merged_df = pd.merge(merged_df, ads_df, how='inner',
                             left_on=matching_columns, right_on=matching_columns,
                             suffixes=('', f'_{ads}'))

    duplicated_cols = [name.strip("_" + ads_list[-1])
                       for name in merged_df.columns
                       if "_" + ads_list[-1] in name]

    merged_df = merged_df.rename(columns={col: col + "_" + ads_list[0] for col in duplicated_cols})

    return merged_df

def compute_oer_eta(df):
    df = compute_oer_reaction_energies(df)
    df['maxG'] = df[['delG1', 'delG2', 'delG3', 'delG4']].apply(max, axis=1)
    df['RDS'] = df.apply(get_rds, axis=1)
    df['eta'] = df['maxG'] - 1.23
    return df

def compute_oer_reaction_energies(df):
    df['delG1'] = df['adsorption_free_energy_OH']
    df['delG2'] = df['adsorption_free_energy_O'] - df['adsorption_free_energy_OH']
    df['delG3'] = df['adsorption_free_energy_HO2'] - df['adsorption_free_energy_O']
    df['delG4'] = 4.92 - df['adsorption_free_energy_HO2']
    return df

def get_rds(row):
    rn1 = row['delG1']
    rn2 = row['delG2']
    rn3 = row['delG3']
    rn4 = row['delG4']
    maxVal = np.argmax(np.array([rn1, rn2, rn3, rn4]))
    return maxVal+1

def objective1(df, a, b):
    mask = (df['delG2'] >= a) & (df['delG2'] <= b)
    return df[mask]

def plot(intercept, slope, df, x, y):
    y_calc = intercept + slope*np.array(list(df[x]))
    f, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(df[x], df[y], 
                color='k', s=1)
    plt.plot(df[x], y_calc)
    #plt.title('Adsorption Energy in eV', fontsize=24)
    plt.xlabel(x, fontsize=32, fontweight='bold')
    plt.ylabel(y, fontsize=32, fontweight='bold')
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.text(0.99, 0.99, 'intercept = '+ str(round(intercept,2))+', slope = '
              +str(round(slope,2)), fontsize=32, fontweight='bold',
              ha='right', va='top', transform=ax.transAxes)
    plt.show()    
   
def scalingEtaPlot(adE, a, b, c):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(adE['delG2'], -adE['eta'], color='k')
    ax.axvline(x=a, c='r')
    ax.axvline(x=b, c='r')
    ax.axhline(y=c, c='r')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('O-OH /eV', fontsize=28, fontweight='bold') 
    plt.ylabel(r'$-\eta_{TD}$ /V', fontsize=28, fontweight='bold')
    #plt.title('Overpotential vs O-OH difference')  
    return fig

#distribution plot
def plot3(df_oer, a, b, c):
    fig = plt.figure(figsize=(10, 8))
    # Set up the GridSpec layout: 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    # Main plot: bottom-left
    ax_main = fig.add_subplot(gs[1, 0])
    # Top plot: top-left, shares x with main
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right plot: bottom-right, shares y with main
    ax_right = fig.add_subplot(gs[1, 1])
    # Axis
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_top.get_yticklabels(), visible=False)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_top.tick_params(axis='y', which='both', left=False, right=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_right.tick_params(axis='y', which='both', left=False, right=False)
    ax_main.tick_params(labelsize=20)
    # Plots 
    ax_main.scatter(df_oer['delG2'], -df_oer['eta'], color='k', label='volcano plot')
    # ax_main.axvline(x=a, c='r')
    # ax_main.axvline(x=b, c='r')
    # ax_main.axhline(y=c, c='r')
    ax_top.hist(df_oer['delG2'], color='gray', bins=150)         # top plot
    ax_right.hist(df_oer['eta'], color='orange', bins=100, orientation='horizontal')    # right plot
    ax_right.invert_yaxis()
    # Titles 
    ax_main.set_xlabel(r"$AE_O-AE_{OH}$ /eV", fontsize=28)
    ax_main.set_ylabel(r'$-\eta_{TD}$ /V', fontsize=28)
    ax_top.set_title(r"$AE_O-AE_{OH}$ distribution", fontsize=28)
    ax_right.set_title(r'$-\eta_{TD}$ distribution', fontsize=28, rotation=-90, x=1.2, y=0.2)
    plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
    plt.show()
  
#distribution plot2
def plot4(df_oer, a, b, c):
    fig = plt.figure(figsize=(10, 8))
    # Set up the GridSpec layout: 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    # Main plot: bottom-left
    ax_main = fig.add_subplot(gs[1, 0])
    # Top plot: top-left, shares x with main
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right plot: bottom-right, shares y with main
    ax_right = fig.add_subplot(gs[1, 1])
    # Axis
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_top.get_yticklabels(), visible=False)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_top.tick_params(axis='y', which='both', left=False, right=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_right.tick_params(axis='y', which='both', left=False, right=False)
    ax_main.tick_params(labelsize=20)
    # Plots 
    ax_main.scatter(df_oer['delG2'], -df_oer['eta'], color='k', label='volcano plot')
    # ax_main.axvline(x=a, c='r')
    # ax_main.axvline(x=b, c='r')
    # ax_main.axhline(y=c, c='r')
    ax_top.hist(df_oer['delG2'], color='gray', bins=150)         # top plot
    ax_right.hist(df_oer['maxG'], color='orange', bins=100, orientation='horizontal')    # right plot
    ax_right.invert_yaxis()
    # Titles 
    ax_main.set_xlabel(r"$AE_{O}-AE_{OH}$ /eV", fontsize=28)
    ax_main.set_ylabel(r'$-\eta_{TD}$ /V', fontsize=28)
    ax_top.set_title(r"$AE_{O}-AE_{OH}$ distribution", fontsize=28)
    ax_right.set_title(r'$\Delta_{r}G_{max}$ distribution', fontsize=28, rotation=-90, x=1.2, y=0.2)
    plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
    plt.show()  


