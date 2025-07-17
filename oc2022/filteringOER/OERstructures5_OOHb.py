#run from: /mnt/c/users/Admin/Desktop/ML/oc2022/
#for HO2's used for OER analysis
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

filename = '/mnt/c/users/Admin/Desktop/ML/oc2022/all3.csv' 
adE = pd.read_csv(filename) 

#OOH search: creates a list of the locations of all OOH systems 
OOH_list = list(adE['system_id_HO2'].unique())
OOH_set = set(OOH_list)
adE2 = adE.copy()
del adE2['nads'] #not sure 'nads' is for which adsorbate, delete this column

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
numbers.to_csv('OERnumbersOOH.csv')

breakpoint()

#Calculate OOH bond lengths and angles
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
     OO = atoms.get_distance(O1, O2)
     angle1 = atoms.get_angle(O2, O1, H1) #O1 is the central atom next to H
     angle2 = atoms.get_angle(O1, O2, H1) #O2 is the central atom next to H
     OH1 = atoms.get_distance(O1, H1)
     OH2 = atoms.get_distance(O2, H1)
     results[sid[i]]= {'filename': file[i], 'entry number': entry[i], 
                    'composition': atoms.get_chemical_formula(), 'nads': data.nads,
                    'OH_length 1': OH1, 'OH_length 2': OH2,
                    'OO_length': OO, 'OOH angle 1': angle1, 'OOH angle 2': angle2 }
     if data.nads >= 2:
        H2 = H_pos[1]
        O3 = ads_pos[2]; O4 = ads_pos[3]
        OO_b = atoms.get_distance(O3, O4)
        angle1_b = atoms.get_angle(O3, O4, H2) 
        angle2_b = atoms.get_angle(O4, O3, H2) 
        OH1_b = atoms.get_distance(O3, H2)
        OH2_b = atoms.get_distance(O4, H2) 
        results2[sid[i]]= {'OH_length 1b': OH1_b, 'OH_length 2b': OH2_b,
                    'OO_lengthb': OO_b, 'OOH angle 1b': angle1_b, 'OOH angle 2b': angle2_b }
        print(sid[i], 'filename', file[i], 'entry number', entry[i]) #check manually later  
     
results_df = pd.DataFrame.from_dict(results, orient='index')  
results2_df = pd.DataFrame.from_dict(results2, orient='index') #nads > 1, 2nd adsorbate
df = pd.concat([results_df, results2_df], axis=1) #412

breakpoint()
adE2 = adE2.set_index('system_id_HO2')
adE3 = df.join(adE2, how='inner') #491, cross-check here
adE3.to_csv('OERangles_OOH.csv') #491
adE3 = adE3[~adE3.index.duplicated(keep='first')] #412

#on checking manually it was found that all the entries below had at least one OOH with incorrect geometry
# and hence were not included
#nads=2
# 41687 filename is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0012.lmdb entry number 573
# 41774 filename is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0028.lmdb entry number 1053
#nads = 3: some OOH's are ok, some are not; all 3 should not not included after filtering
# 44497 filename is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0034.lmdb entry number 183

#ex: sid 68776
#check that the 'bulk_symbols', atoms.get_chemical_formula() match
#stoichiometry may not be exact
dataset = LmdbDataset({'src': 'is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0000.lmdb'})
data = dataset[170] 
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
write_vasp(atoms.get_chemical_formula()+'_HO2_CONTCAR', atoms)




