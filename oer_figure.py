import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import oc_analyzer.oc2022.oer_utils as oer_functions2

filename = 'data/oc2022/adsorption_energies.csv' #all adsorbates: 31425
adE = pd.read_csv(filename, index_col=0, na_values='')
#adE = adE.dropna() #31425
#adE.drop(['traj_id', 'y_relaxed', 'natoms', 'nads2'], axis=1, inplace=True)
#adE.reset_index(inplace=True)

mask = (adE['ads_symbols'] == 'H2O') | (adE['ads_symbols'] == 'OH') | (adE['ads_symbols'] == 'O') | (adE['ads_symbols'] == 'HO2') | (adE['ads_symbols'] == 'O2') 
adE2 = adE[mask]  #no correction to O, OH, OOH at this point, 19255

#corrections 
adE3 = adE2.copy()
adE3['corrected_adsorption_energy'] = adE3.apply(oer_functions2.corr, axis=1)

hue_order =['H2O', 'OH', 'O', 'HO2',  'O2']

del adE3['adsorption_energy']
adE3 = adE3.rename(columns={'corrected_adsorption_energy': 'adsorption_energy'})
#adE3.to_csv('correctedAdE.csv')

#distributions
#supporting figures
sns.set(font_scale=3, rc={'font.weight': 'bold'})
sns.set_style('ticks')
f, ax = plt.subplots(figsize=(10, 12))
sns.histplot(y=adE3['ads_symbols'], color='green')
ax.set_xlabel('Adsorbate', fontsize=32, fontweight='bold')
ax.set_ylabel('Count', fontsize=32, fontweight='bold')
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
plt.show()

sns.set(font_scale=2, rc={'font.weight': 'bold'})
sns.set_style('ticks')
f, ax = plt.subplots(figsize=(10, 10))
hue_order =['H2O', 'OH', 'O', 'HO2',  'O2']
sns.violinplot(y=adE3['adsorption_energy'],hue=adE3['ads_symbols'], hue_order=hue_order, palette=sns.color_palette('dark'), fill=False, split=True, inner='quart')
#ax.set_xlabel('Distribution', fontsize=32, fontweight='bold')
ax.set_ylabel('Adsorption Energy (eV)', fontsize=32, fontweight='bold')
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
plt.show()

# #ranges
# adE_OH = oer_functions2.maskAndSort(adE3, 'OH') 
# print(len(adE_OH), adE_OH['adsorption_energy'].min(), adE_OH['adsorption_energy'].max(), adE_OH['adsorption_energy'].mean())
# adE_O = oer_functions2.maskAndSort(adE3, 'O') 
# print(len(adE_O), adE_O['adsorption_energy'].min(), adE_O['adsorption_energy'].max(), adE_O['adsorption_energy'].mean())
# adE_HO2 = oer_functions2.maskAndSort(adE3, 'HO2') 
# print(len(adE_HO2), adE_HO2['adsorption_energy'].min(), adE_HO2['adsorption_energy'].max(), adE_HO2['adsorption_energy'].mean())

#merged dataframe with surfaces with all 3 adsorbates: OH, O, HO2
df_oer = oer_functions2.mergeThree(adE3, 'OH', 'O', 'HO2')

# #range
# print('OH: ', df_oer['OH'].min(), df_oer['OH'].max(), df_oer['OH'].mean())
# print('O: ', df_oer['O'].min(), df_oer['O'].max(), df_oer['O'].mean())
# print('OOH: ', df_oer['HO2'].min(), df_oer['HO2'].max(), df_oer['HO2'].mean())

#THERMODYNAMIC OVERPOTENTIAL
df_oer = oer_functions2.reaction_energies(df_oer)
df_oer['maxG'] = df_oer[['delG1', 'delG2', 'delG3', 'delG4']].apply(max, axis=1)
df_oer['RDS'] = df_oer.apply(oer_functions2.get_rds, axis=1)
df_oer['eta'] = df_oer['maxG'] - 1.23
df_oer2 = df_oer.copy().sort_values(by='eta')
#df_oer2.to_csv('all3.csv')

oer_functions2.scalingEtaPlot(df_oer, 1.24, 1.64, -0.4)
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
#plt.show()


#plot
mask = (df_oer['delG2'] >= -5 ) & (df_oer['delG2'] <= 5 )
df_oer2 = df_oer[mask]
oer_functions2.plot3(df_oer2, 1.2, 1.5, -0.4)
#oer_functions.plot4(df_oer, 1.2, 1.5, -0.4)

plt.show()

# mask = df_oer['eta'] < 0.4
# good_catalysts = df_oer[mask]
# good_catalysts = good_catalysts.sort_values(by='eta')
# print(good_catalysts[['bulk_symbols', 'bulk_id', 'eta']])
# #good_catalysts.to_csv('good.csv')

# mask2 = df_oer['eta'] < 0.5
# good_cat2 = df_oer[mask2]
# good_cat2 = good_cat2.sort_values(by='eta')
# print(good_cat2[['bulk_symbols', 'bulk_id', 'eta']])

