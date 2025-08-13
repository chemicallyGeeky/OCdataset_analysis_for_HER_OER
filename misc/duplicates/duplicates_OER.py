import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import newPlots
import seaborn as sns
import oer_functions2
import matplotlib.gridspec as gridspec

filename = 'data/oc2022/adsorption_energies.csv' #all adsorbates: 31425
adE = pd.read_csv(filename, index_col=0, na_values='')
adE.drop(['traj_id', 'y_relaxed', 'natoms', 'nads2'], axis=1, inplace=True)

maskOH =  (adE['ads_symbols'] == 'OH')  
maskO = (adE['ads_symbols'] == 'O')  
maskOOH = (adE['ads_symbols'] == 'HO2')

adE_OH = adE[maskOH] #3060
adE_O = adE[maskO] #7996
adE_OOH = adE[maskOOH] #3238

#analysis 1: same bulk_id, miller_index, slab_sid, nads: AE difference from different sites
#analysis 2: same bulk_id, miller_index, slab_sid, but different nads: AE difference from 
# different coverages and optionally sites
#analysis 3: same bulk_id, miller_index but different slab_sid; nads may be same or different;
#AE difference from surface terminations, coverages or sites

#ADSORBATE OH
adE_OH = adE_OH.sort_values(by=['bulk_id', 'miller_index', 'slab_sid', 'nads'])
#analysis 1: OH
OH_duplicates1 = adE_OH[adE_OH.duplicated(subset=['bulk_id', 'miller_index', 'slab_sid', 'nads'], keep=False)] #28

OH_duplicates1['diff'] = OH_duplicates1.groupby(['bulk_id', 'miller_index', 'slab_sid', 'nads'])['adsorption_energy'].diff().abs()
print('Adsorbate OH')
print('same bulk_id, miller_index, slab_sid, nads:')
print('min, mean, max: ', OH_duplicates1['diff'].min(), OH_duplicates1['diff'].mean(), OH_duplicates1['diff'].max())

adE_OH1 = adE_OH.drop(OH_duplicates1.index)

#analysis 2: OH
OH_with_diff_nads = (
    adE_OH1.groupby(['bulk_id', 'miller_index', 'slab_sid'])
    ['nads'].nunique().reset_index(name='nads_count'))

OH_with_diff_nads = OH_with_diff_nads[OH_with_diff_nads['nads_count'] > 1] #12

OH_duplicates2 = adE_OH1.reset_index().merge(
    OH_with_diff_nads.drop(columns='nads_count'),
    on=['bulk_id', 'miller_index', 'slab_sid'],
    how='inner')

OH_duplicates2['diff'] = OH_duplicates2.groupby(['bulk_id', 'miller_index', 'slab_sid'])['adsorption_energy'].diff().abs()
print('same bulk_id, miller_index, slab_sid:')
print('min, mean, max: ', OH_duplicates2['diff'].min(), OH_duplicates2['diff'].mean(), OH_duplicates2['diff'].max())

OH_duplicates2.set_index('system_id', inplace=True)
adE_OH2 = adE_OH1.drop(OH_duplicates2.index)

#analysis 3: OH
OH_with_diff_surf = (
    adE_OH2.groupby(['bulk_id', 'miller_index'])
    ['slab_sid'].nunique().reset_index(name='slab_count'))

OH_with_diff_surf = OH_with_diff_surf[OH_with_diff_surf['slab_count'] > 1] #319

OH_duplicates3 = adE_OH2.reset_index().merge(
    OH_with_diff_surf.drop(columns='slab_count'),
    on=['bulk_id', 'miller_index'],
    how='inner')

OH_duplicates3['diff'] = OH_duplicates3.groupby(['bulk_id', 'miller_index'])['adsorption_energy'].diff().abs()
OH_duplicates3.set_index('system_id', inplace=True)
adE_OH3 = adE_OH2.drop(OH_duplicates3.index)

print('same bulk_id, miller_index')
print('min, mean, max: ', OH_duplicates3['diff'].min(), OH_duplicates3['diff'].mean(), OH_duplicates3['diff'].max())

breakpoint()

#ANALYSIS O
adE_O = adE_O.sort_values(by=['bulk_id', 'miller_index', 'slab_sid', 'nads'])
#analysis 1: O
O_duplicates1 = adE_O[adE_O.duplicated(subset=['bulk_id', 'miller_index', 'slab_sid', 'nads'], keep=False)] 

O_duplicates1['diff'] = O_duplicates1.groupby(['bulk_id', 'miller_index', 'slab_sid', 'nads'])['adsorption_energy'].diff().abs()
print('Adsorbate O')
print('same bulk_id, miller_index, slab_sid, nads:')
print('min, mean, max: ', O_duplicates1['diff'].min(), O_duplicates1['diff'].mean(), O_duplicates1['diff'].max())

adE_O1 = adE_O.drop(O_duplicates1.index)

#analysis 2: O
O_with_diff_nads = (
    adE_O1.groupby(['bulk_id', 'miller_index', 'slab_sid'])
    ['nads'].nunique().reset_index(name='nads_count'))

O_with_diff_nads = O_with_diff_nads[O_with_diff_nads['nads_count'] > 1] 

O_duplicates2 = adE_O1.reset_index().merge(
    O_with_diff_nads.drop(columns='nads_count'),
    on=['bulk_id', 'miller_index', 'slab_sid'],
    how='inner')

O_duplicates2['diff'] = O_duplicates2.groupby(['bulk_id', 'miller_index', 'slab_sid'])['adsorption_energy'].diff().abs()
print('same bulk_id, miller_index, slab_sid:')
print('min, mean, max: ', O_duplicates2['diff'].min(), O_duplicates2['diff'].mean(), O_duplicates2['diff'].max())

O_duplicates2.set_index('system_id', inplace=True)
adE_O2 = adE_O1.drop(O_duplicates2.index)

#analysis 3: O
O_with_diff_surf = (
    adE_O2.groupby(['bulk_id', 'miller_index'])
    ['slab_sid'].nunique().reset_index(name='slab_count'))

O_with_diff_surf = O_with_diff_surf[O_with_diff_surf['slab_count'] > 1] 

O_duplicates3 = adE_O2.reset_index().merge(
    O_with_diff_surf.drop(columns='slab_count'),
    on=['bulk_id', 'miller_index'],
    how='inner')

O_duplicates3['diff'] = O_duplicates3.groupby(['bulk_id', 'miller_index'])['adsorption_energy'].diff().abs()
O_duplicates3.set_index('system_id', inplace=True)
adE_O3 = adE_O2.drop(O_duplicates3.index)

print('same bulk_id, miller_index')
print('min, mean, max: ', O_duplicates3['diff'].min(), O_duplicates3['diff'].mean(), O_duplicates3['diff'].max())
breakpoint()

#ANALYSIS OOH

adE_OH = adE_OH.sort_values(by=['bulk_id', 'miller_index', 'slab_sid', 'nads'])
#analysis 1: OOH
OOH_duplicates1 = adE_OOH[adE_OOH.duplicated(subset=['bulk_id', 'miller_index', 'slab_sid', 'nads'], keep=False)] 

OOH_duplicates1['diff'] = OOH_duplicates1.groupby(['bulk_id', 'miller_index', 'slab_sid', 'nads'])['adsorption_energy'].diff().abs()
print('ADSORBATE OOH')
print('same bulk_id, miller_index, slab_sid, nads:')
print('min, mean, max: ', OOH_duplicates1['diff'].min(), OOH_duplicates1['diff'].mean(), OOH_duplicates1['diff'].max())

adE_OOH1 = adE_OOH.drop(OOH_duplicates1.index)

#analysis 2: OOH
OOH_with_diff_nads = (
    adE_OOH1.groupby(['bulk_id', 'miller_index', 'slab_sid'])
    ['nads'].nunique().reset_index(name='nads_count'))

OOH_with_diff_nads = OOH_with_diff_nads[OOH_with_diff_nads['nads_count'] > 1] #12
print('same bulk_id, miller_index, slab_sid: ', len(OOH_with_diff_nads))
#empty
# OOH_duplicates2 = adE_OOH1.reset_index().merge(
#     OOH_with_diff_nads.drop(columns='nads_count'),
#     on=['bulk_id', 'miller_index', 'slab_sid'],
#     how='inner')

# OOH_duplicates2['diff'] = OOH_duplicates2.groupby(['bulk_id', 'miller_index', 'slab_sid'])['adsorption_energy'].diff().abs()
# print('same bulk_id, miller_index, slab_sid:')
# print('min, mean, max: ', OOH_duplicates2['diff'].min(), OOH_duplicates2['diff'].mean(), OOH_duplicates2['diff'].max())

# OOH_duplicates2.set_index('system_id', inplace=True)
# adE_OOH2 = adE_OOH1.drop(OOH_duplicates2.index)

#analysis 3
OOH_with_diff_surf = (
    adE_OOH1.groupby(['bulk_id', 'miller_index'])
    ['slab_sid'].nunique().reset_index(name='slab_count'))

OOH_with_diff_surf = OOH_with_diff_surf[OOH_with_diff_surf['slab_count'] > 1] #319

OOH_duplicates3 = adE_OOH1.reset_index().merge(
    OOH_with_diff_surf.drop(columns='slab_count'),
    on=['bulk_id', 'miller_index'],
    how='inner')

OOH_duplicates3['diff'] = OOH_duplicates3.groupby(['bulk_id', 'miller_index'])['adsorption_energy'].diff().abs()
OOH_duplicates3.set_index('system_id', inplace=True)
adE_OOH3 = adE_OOH1.drop(OOH_duplicates3.index)

print('same bulk_id, miller_index')
print('min, mean, max: ', OOH_duplicates3['diff'].min(), OOH_duplicates3['diff'].mean(), OOH_duplicates3['diff'].max())
breakpoint()

