#run with corrected AEs
#modified code to search through structure
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
plt.rcParams['mathtext.default'] = 'regular'

corrOH=0.26
corrO=-0.03
corrHO2=0.22
def corr(row):
    ads = row['ads_symbols']
    energy = row['adsorption_energy']
    #num = row['nads']
    num = 1
    if ads == 'OH':
        energy += num*corrOH
    if ads == 'O':
        energy += num*corrO
    if ads == 'HO2':
        energy += num*corrHO2        
    return energy  

def maskAndSort (df, ads):
    mask = (df['ads_symbols'] == ads) 
    df_ads = df[mask]
    del df_ads['ads_symbols']
    df_ads = df_ads.set_index('slab_sid') #done at this point as same slab_sid may have multiple adsorbates
    df_ads.sort_index(inplace=True)
    return df_ads

def scaling(x_vals, y_vals):
    reg = LinearRegression()
    reg.fit(x_vals.reshape(-1, 1), y_vals)
    return reg.intercept_, reg.coef_[0]

def twoAdsorbates(adE, x, y):
    df_x = maskAndSort(adE, x)
    df_y = maskAndSort(adE, y)
    df_both = pd.merge(df_x, df_y, how='inner', left_index=True, right_index=True)
    x_vals = np.array(list(df_both['adsorption_energy_x']))
    y_vals = np.array(list(df_both['adsorption_energy_y']))
    scaling_intercept, scaling_slope = scaling(x_vals, y_vals)
    return scaling_intercept, scaling_slope, df_x, df_y, df_both

def mergeThree(adE, x, y, z):
    df_x = maskAndSort(adE, x)
    df_y = maskAndSort(adE, y)
    df_z = maskAndSort(adE, z)
    df_both = pd.merge(df_x, df_y, how='inner', left_index=True, right_index=True)
    df_three = pd.merge(df_both, df_z, how='inner', left_index=True, right_index=True)
    #scaling
    x_vals = np.array(list(df_three['adsorption_energy_x']))
    y_vals = np.array(list(df_three['adsorption_energy_y']))
    z_vals = np.array(list(df_three['adsorption_energy']))
    scaling_intercept_xy, scaling_slope_xy = scaling(x_vals, y_vals)
    print('slope-' + x + '-' + y +': ', scaling_slope_xy, ' intercept-'+ x + '-'+ y+': ', scaling_intercept_xy)
    scaling_intercept_xz, scaling_slope_xz = scaling(x_vals, z_vals)
    print('slope-' + x + '-' + z +': ', scaling_slope_xz, ' intercept-'+ x + '-'+ z+': ', scaling_intercept_xz)
    scaling_intercept_yz, scaling_slope_yz = scaling(y_vals, z_vals)
    print('slope-' + y + '-' + z +': ', scaling_slope_yz, ' intercept-'+ y + '-'+ z+': ', scaling_intercept_yz)
    df_three = df_three.rename(columns={'adsorption_energy_x': x, 'adsorption_energy_y': y, 'adsorption_energy': z })
    df_three = df_three.rename(columns={'system_id_x': 'system_id_'+x, 'system_id_y': 'system_id_'+y, 'system_id': 'system_id_'+z })
    df_three.drop(['bulk_id_x', 'miller_index_x', 'nads_x', 'bulk_symbols_x', 'bulk_id_y', 'miller_index_y', 'nads_y', 'bulk_symbols_y'], axis=1, inplace=True)
    return df_three

def reaction_energies(adE):
    adE['delG1'] = adE['OH']
    adE['delG2'] = adE['O'] - adE['OH']
    adE['delG3'] = adE['HO2'] - adE['O']
    adE['delG4'] = 4.92 - adE['HO2']
    return adE

def overpotential(delG1, delG2, delG3, delG4):
    return (max(delG1, delG2, delG3, delG4))

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


