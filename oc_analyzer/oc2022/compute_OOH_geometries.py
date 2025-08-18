#USE: /mnt/c/users/Admin/Desktop/ML/oc2022/
#search all structures containing HO2
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

filename = '/mnt/c/users/Admin/Desktop/ML/oc2022/correctedAdE.csv' 
adE = pd.read_csv(filename)

mask = adE['ads_symbols'] == 'HO2'
adE2 = adE[mask]
adE2 = adE2.set_index('system_id')
#OOH search
OOH_list = list(adE2.index)
OOH_set = set(OOH_list)

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

mask1 = adE2['nads']==1 #3214
mask2 = adE2['nads']==2 #21
mask3 = adE2['nads']==3 #2
mask4 = adE2['nads']==4 #1
print(len(adE2[mask1]), len(adE2[mask2]), len(adE2[mask3]), len(adE2[mask4]))
breakpoint()

#OOH bond lengths and angles
results = {} #all first OOH
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
     
results_df = pd.DataFrame.from_dict(results, orient='index')  
results2_df = pd.DataFrame.from_dict(results2, orient='index') #nads > 1, 2nd adsorbate
df = pd.concat([results_df, results2_df], axis=1) #combine

breakpoint()
adE2 = adE2.rename(columns={'nads': 'nads0'})
adE3 = df.join(adE2, how='inner')
adE3.to_csv('angles_OOH.csv') #3238

#nads > 2
# 43116 filename is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0017.lmdb entry number 150
# 42932 filename is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0033.lmdb entry number 573
# 44497 filename is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0034.lmdb entry number 183

#ex 1: 4 adsorbates
dataset = LmdbDataset({'src': 'is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0034.lmdb'})
data = dataset[183] 
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




