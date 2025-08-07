#stability analysis
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import newPlots
import seaborn as sns
import oer_functions2
import matplotlib.gridspec as gridspec

def distribution_plot(df, bins=25):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.hist(df['decomposition_energy'], bins=bins, color='k')
    plt.xlabel('decomposition_energy /eV', fontsize=32, fontweight='bold')
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
    plt.show()

def decompEnergy_eta_plot(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(df['decomposition_energy'], df['eta'], color='k', s=5)
    plt.xlabel('decomposition_energy /eV', fontsize=32, fontweight='bold')
    plt.ylabel(r'$\eta_{TD}$ /V', fontsize=32, fontweight='bold')
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    ax.axvspan(xmin=ax.get_xlim()[0], xmax=0.1, color='lightgray', alpha=0.5)
    plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
    plt.show()

#pourbaix energy data
filename = 'pourbaix_v1.csv' 
decomp_df_orig = pd.read_csv(filename, na_values='') #722 entries

mask = decomp_df_orig['voltage'] == 1.63
decomp_df = decomp_df_orig[mask]
decomp_df = decomp_df.rename(columns={'material_id':'bulk_id'})
decomp_df = decomp_df.set_index('bulk_id') #305 entries

#how many are stable
stability_mask = decomp_df['decomposition_energy'] <= 0.1
stable_df = decomp_df[stability_mask] #27 entries
print(stable_df[['name', 'decomposition_energy', 'energy_above_hull']])

#version1: unfiltered
filename_unfiltered = 'all3.csv' 
etaUnfiltered = pd.read_csv(filename_unfiltered, index_col=0, na_values='')
etaUnfiltered = etaUnfiltered.set_index('bulk_id') #491 entries
mergedUnfiltered = decomp_df.join(etaUnfiltered, how='inner')
print(len(mergedUnfiltered)) #417

#plot 1: distribution
distribution_plot(mergedUnfiltered)
breakpoint()
#plot 2
decompEnergy_eta_plot(mergedUnfiltered)
breakpoint()

#version 2:  filtered (unphysical structures removed)
filename_filtered = 'oer_filtered_sorted.csv' 
etaFiltered = pd.read_csv(filename_filtered, index_col=0, na_values='')
etaFiltered = etaFiltered.set_index('bulk_id') #92
merged_filtered = decomp_df.join(etaFiltered, how='inner') #74

#plot 1
distribution_plot(merged_filtered)
breakpoint()
#plot 2
decompEnergy_eta_plot(merged_filtered)
breakpoint()

stability_mask2 = merged_filtered['decomposition_energy'] <= 0.1
breakpoint()
