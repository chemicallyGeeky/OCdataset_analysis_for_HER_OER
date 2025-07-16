#run from containing is2res_total_train_val_test_lmdbs
# CORRECT, FINAL FOR DATA EXTRACTION from LmdbDataset and metadata
import pickle 
import numpy as np
import pandas as pd
from glob import glob
from fairchem.core.datasets import LmdbDataset

#extract system_id and y_relaxed, natoms, n_ads from the Lmdb dataset
#saved in dicts with system_id/sid as the dict key
filename_dict = {}
systemId_dict = {}; energies_dict = {}; totalAtoms_dict = {}; totalAds_dict = {}
for filename in (glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.*.lmdb") + 
                glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_id/data.*.lmdb") +
                glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_ood/data.*.lmdb")):
     for d in LmdbDataset({'src':filename}):
          filename_dict[d['sid']] = d
          systemId_dict[d['sid']] = d['sid']
          energies_dict[d['sid']] = d['y_relaxed']
          totalAtoms_dict[d['sid']] = d['natoms']
          totalAds_dict[d['sid']] = d['nads']

#combine to dataframe and export
allEnergies = pd.DataFrame([[*systemId_dict.values()], [*energies_dict.values()],[*totalAtoms_dict.values()], [*totalAds_dict.values()]],index=['system_id', 'y_relaxed', 'natoms', 'nads2']).T
allEnergies = allEnergies.astype({'system_id': int, 'natoms': int, 'nads2': int})
allEnergies = allEnergies.set_index('system_id')
allEnergies = allEnergies.sort_index()

allEnergies.to_csv('fromLmdbData.csv')

#CHECK: load a Lmdb Dataset and check 
breakpoint()
a = LmdbDataset({"src":"is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0029.lmdb"})
print(a[0])
#read its y_relaxed and sid: energies[sid] should match its y_relaxed value

#load pickle file
mdata = pickle.load(open("oc22_metadata.pkl", "rb")) #pickle file must be in folder
#The keys of this dictionary are the system-ids
system_id = list(mdata.keys())

#'bulk_id', miller_index':, 'nads', 'traj_id', 'bulk_symbols', 'slab_sid', 'ads_symbols'
s = {}
for i in system_id:
    try:
        s[i] = pd.concat([pd.Series({'system_id':i}),pd.Series(mdata[i])])
    except KeyError:
        s[i] = pd.Series({'system_id': i,'bulk_id': np.NaN, 'miller_index': np.NaN, 'nads': np.NaN, 'traj_id': np.NaN,
                          'bulk_symbols': np.NaN, 'slab_sid': np.NaN, 'ads_symbols': np.NaN })  
        
column_names = ['system_id','bulk_id', 'miller_index', 'nads', 'traj_id', 'bulk_symbols', 'slab_sid', 'ads_symbols']
cols = {}
for j in range(len(column_names)):
        cols[j] = []
        for i in system_id: #i in range(len(s)) also works but am not sure whether that covers all system_id's: it is not
            try:
                cols[j].append(s[i][column_names[j]])
            except KeyError:
                cols[j].append(np.NaN)
#CHECK here that s[i] and mdata[i] match
breakpoint()

#create dataframe
df0 = pd.DataFrame([cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]], 
                  index=['system_id','bulk_id', 'miller_index', 'nads', 'traj_id', 'bulk_symbols', 'slab_sid', 'ads_symbols' ]).T
#clean 
df = df0.set_index('system_id')
df = df.sort_index()
print('unique compositions = ', df0['bulk_symbols'].nunique(), 'unique surfaces = ', df0['miller_index'].nunique())
print(' unique mp ids = ', df0['bulk_id'].nunique())

#CHECK: mdata[i], s[i], df.loc[i] match. 
breakpoint()

#this may not be required
df = df.astype({'bulk_id': 'str', 'miller_index': 'str', 'traj_id': 'str', 'ads_symbols':'str'})
names = ['bulk_id', 'miller_index', 'traj_id', 'ads_symbols']
for name in names:
    df[name] = [x.strip() for x in df[name]]
df.to_csv('fromMetadata.csv')

#all
allE = pd.merge(df, allEnergies, how='inner', left_index=True, right_index=True)
print('Check whether the two nads are equal: answer should be 0 ' , np.sum(allE['nads']!=allE['nads2'])) #should be 0
allE.to_csv('lmdbPlusMetaData.csv')

#CHECK: find system_id value =sid from system_id[i] & mdata[i]. Should match allEnergies[sid] and allE[sid]
breakpoint()