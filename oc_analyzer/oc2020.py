#run from folder with dataset folders
#source catalyst/bin/activate
import pickle 
import numpy as np
import pandas as pd
from glob import glob, iglob
from fairchem.core.datasets import LmdbDataset

#extract system_id and y_relaxed, natoms, n_ads from the Lmdb dataset
#saved in dicts with system_id/sid as the dict key
filename_dict = {}
systemId_dict = {}; energies_dict = {}; totalAtoms_dict = {}; totalAds_dict = {}
for filename in iglob("is2res_train_val_test_lmdbs/data/is2re/100k/train/data.lmdb"):
      for d in LmdbDataset({'src':filename}):
          filename_dict[d['sid']] = d
          systemId_dict[d['sid']] = d['sid']
          energies_dict[d['sid']] = d['y_relaxed']
          totalAtoms_dict[d['sid']] = d['natoms']
          
#combine to dataframe and export
allEnergies = pd.DataFrame([[*systemId_dict.values()], [*energies_dict.values()],[*totalAtoms_dict.values()]],index=['system_id', 'y_relaxed', 'natoms']).T
allEnergies = allEnergies.astype({'system_id': int, 'natoms': int})
allEnergies = allEnergies.set_index('system_id')
allEnergies = allEnergies.sort_index()
#CHECK: filename_dict[i] matches allEnergies[i]

allEnergies.to_csv('lmdbData100kFINAL.csv')
#export filename_dict as a pickle file
with open("lmdb100k.pkl", "wb") as f:
    pickle.dump(filename_dict, f)

#convert identifier list to a dataframe and search
#load using pickle file
df = pd.read_csv('fromMetadataFINAL.csv', index_col=0)

#all
allE = pd.merge(df, allEnergies, how='inner', left_index=True, right_index=True)
allE = allE.rename(columns={'y_relaxed':'adsorption_energy'})
allE.to_csv('lmdbPlusMetaDataFINAL.csv')


