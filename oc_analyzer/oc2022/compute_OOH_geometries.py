#USE: /mnt/c/users/Admin/Desktop/ML/oc2022/
#search all structures containing the 'HO2' adsorbate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fairchem.core.datasets import LmdbDataset
from ase import io, Atoms
from ase.visualize import view
from glob import glob, iglob
import os
import re
import pickle
from ase.io.vasp import write_vasp

filename = 'sortedWithAdE.csv' #dataset of adsorption energies 31425 adsorbate-catalyst system
adE = pd.read_csv(filename)

mask = adE['ads_symbols'] == 'HO2' #keep those entries with HO2 adsorbates: 3238 entires
adE2 = adE[mask]
adE2 = adE2.set_index('system_id')
#OOH search
OOH_list = list(adE2.index)
OOH_set = set(OOH_list)

#using the 'system_id' locate where the structural details are in the lmdb datasets
#save this as a dataframe (numbers) and export as a csv file
sid = []; file = []; entry = []
   
for filename in (glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.*.lmdb") + 
                glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_id/data.*.lmdb") +
                glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_ood/data.*.lmdb")):
            dataset = LmdbDataset({'src': filename}) #open the lmdb file
            for i in range(len(dataset)):
                if dataset[i].nads > 0 and dataset[i].sid in OOH_set:
                    #print('filename: ', filename, ' i: ', i, ' sid: ', dataset[i].sid)
                    sid.append(dataset[i].sid); file.append(filename); entry.append(i)
                    OOH_set.remove(dataset[i].sid) 
            if not OOH_set:
                  break          
           
numbers = pd.DataFrame({'filename': file, 'entry number': entry}, index=sid)
numbers.to_csv('numbersOOH.csv')

#number of adsorbates
mask1 = adE2['nads']==1 #3214
mask2 = adE2['nads']==2 #21
mask3 = adE2['nads']==3 #2
mask4 = adE2['nads']==4 #1
print(len(adE2[mask1]), len(adE2[mask2]), len(adE2[mask3]), len(adE2[mask4]))
breakpoint()

#extract OOH bond lengths and angles of each structure
#since we do not know which is the 'central' O of OOH, there are two OH lengths (with either O)
#and two OOH angles (with either O at the vertex)
#only the angles associated with the first two adsorbates were extracted
results = {} #all first OOH adsorbate
results2 = {} # nads > 1
for i in range(len(file)):
     dataset = LmdbDataset({'src': file[i]})
     j = entry[i]
     data = dataset[j]
     cell = data.cell.squeeze(0).numpy() 
     atoms = Atoms( numbers=data.atomic_numbers.tolist(), positions=data.pos_relaxed.numpy(), cell=cell,  pbc=True)
     tags = data.tags.numpy() #for identifying adsorbates, all adsorbate atoms have tag 2
     atomic_numbers = data.atomic_numbers.numpy()
     #get adsorbate sites
     ads_pos = np.where(tags==2)[0]
     H_pos = np.where(atomic_numbers  == 1)[0] #array of 2
     O_pos = np.setdiff1d(ads_pos, H_pos)
     O1 = O_pos[0]; O2 = O_pos[1]
     H1 = H_pos[0]
     OO = atoms.get_distance(O1, O2, mic=True)
     angle1 = atoms.get_angle(O2, O1, H1, mic=True) #O1 is the central atom next to H
     angle2 = atoms.get_angle(O1, O2, H1, mic=True) #O2 is the central atom next to H
     OH1 = atoms.get_distance(O1, H1, mic=True)
     OH2 = atoms.get_distance(O2, H1, mic=True)
     results[sid[i]]= {'filename': file[i], 'entry number': entry[i], 
                    'composition': atoms.get_chemical_formula(), 'nads': data.nads,
                    'OH_length_1': OH1, 'OH_length_2': OH2,
                    'OO_length': OO, 'OOH_angle_1': angle1, 'OOH_angle_2': angle2 }
     if data.nads >= 2:
        H2 = H_pos[1]
        O3 = O_pos[2]; O4 = O_pos[3]
        OO_b = atoms.get_distance(O3, O4, mic=True)
        angle1_b = atoms.get_angle(O4, O3, H2, mic=True) #O3 is the central atom next to H2
        angle2_b = atoms.get_angle(O3, O4, H2, mic=True) #O4 is the central atom next to H2
        OH1_b = atoms.get_distance(O3, H2, mic=True)
        OH2_b = atoms.get_distance(O4, H2, mic=True) 
        results2[sid[i]]= {'OH_length_1b': OH1_b, 'OH_length_2b': OH2_b,
                    'OO_lengthb': OO_b, 'OOH_angle_1b': angle1_b, 'OOH_angle_2b': angle2_b }
        print(sid[i], 'filename', file[i], 'entry number', entry[i]) #check manually later  
     
results_df = pd.DataFrame.from_dict(results, orient='index')  # len 3238
results2_df = pd.DataFrame.from_dict(results2, orient='index') #nads > 1, 2nd adsorbate, len 24
df = pd.concat([results_df, results2_df], axis=1) #combine

breakpoint()
adE2 = adE2.rename(columns={'nads': 'nads0'})
adE_OOH = df.join(adE2, how='inner')

#collapse
cols = ['filename', 'entry number', 'composition', 'nads', 'Unnamed: 0', 'bulk_id', 'miller_index', 'nads0',
       'bulk_symbols', 'slab_sid', 'ads_symbols', 'adsorption_energy']
adE_OOH = adE_OOH.drop(columns=cols, errors='ignore')
adE_OOH['OH length 1'] = adE_OOH.apply(lambda row: [row['OH_length_1']] + ([row['OH_length_1b']] if not pd.isna(row['OH_length_1b']) else []), axis=1)
adE_OOH['OH length 2'] = adE_OOH.apply(lambda row: [row['OH_length_2']] + ([row['OH_length_2b']] if not pd.isna(row['OH_length_2b']) else []), axis=1)

adE_OOH['OOH angle 1'] = adE_OOH.apply(lambda row: [row['OOH_angle_1']] + ([row['OOH_angle_1b']] if not pd.isna(row['OOH_angle_1b']) else []), axis=1)
adE_OOH['OOH angle 2'] = adE_OOH.apply(lambda row: [row['OOH_angle_2']] + ([row['OOH_angle_2b']] if not pd.isna(row['OOH_angle_2b']) else []), axis=1)

adE_OOH['OO length'] = adE_OOH.apply(lambda row: [row['OO_length']] + ([row['OO_lengthb']] if not pd.isna(row['OO_lengthb']) else []), axis=1)

cols = ['traj_id', 'y_relaxed', 'natoms', 'nads2','OH_length_1', 'OH_length_2', 'OO_length', 'OOH_angle_1', 'OOH_angle_2',
       'OH_length_1b', 'OH_length_2b', 'OO_lengthb', 'OOH_angle_1b',
       'OOH_angle_2b']
adE_OOH = adE_OOH.drop(columns=cols, errors='ignore')
adE_OOH.to_csv('OOH_geomteries.csv') #3238

#ex 1: 4 adsorbates
dataset = LmdbDataset({'src': '../is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0000.lmdb'})
data = dataset[624] 
cell = data.cell.squeeze(0).numpy() 
cell = data.cell.squeeze(0).numpy() 
atoms = Atoms( numbers=data.atomic_numbers.tolist(), positions=data.pos_relaxed.numpy(), cell=cell,  pbc=True)
tags = data.tags.numpy()
atomic_numbers = data.atomic_numbers.numpy()
#get adsorbate sites
ads_pos = np.where(tags==2)[0]
H_pos = np.where(atomic_numbers  == 1)[0] #array of 2
O_pos = np.setdiff1d(ads_pos, H_pos)
print(H_pos, O_pos)
print(atoms.get_chemical_formula())
write_vasp('H3Ag24Ba22O48_HO2_CONTCAR', atoms)








