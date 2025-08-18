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

filename = '/mnt/c/users/Admin/Desktop/ML/oc2022/oerFiltered_OOH.csv' 
adE = pd.read_csv(filename, index_col=0)

#OH search
OH_list = list(adE['system_id_OH']) #128
OH_set = set(OH_list)

sid = []; file = []; entry = []
   
for filename in (glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.*.lmdb") + 
                glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_id/data.*.lmdb") +
                glob("is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_ood/data.*.lmdb")):
            dataset = LmdbDataset({'src': filename}) #open the lmdb file
            for i in range(len(dataset)):
                if dataset[i].nads > 0 and dataset[i].sid in OH_set:
                    #print('filename: ', filename, ' i: ', i, ' sid: ', dataset[i].sid)
                    sid.append(dataset[i].sid); file.append(filename); entry.append(i)
                    OH_set.remove(dataset[i].sid) 
            if not OH_set:
                  break          
           
numbers = pd.DataFrame({'filename': file, 'entry number': entry}, index=sid)
numbers.to_csv('OER_numbersOH.csv')

breakpoint()

#OH bond lengths 
results = {} #all first OH
for i in range(len(file)):
     dataset = LmdbDataset({'src': file[i]})
     j = entry[i]
     data = dataset[j]
     cell = data.cell.squeeze(0).numpy() 
     atoms = Atoms( numbers=data.atomic_numbers.tolist(), positions=data.pos_relaxed.numpy(), cell=cell,  pbc=True)
     tags = data.tags.numpy()
     atomic_numbers = data.atomic_numbers.numpy()
     #get adsorbate sites
     ads_pos = np.where(tags==2)[0]
     H_pos = np.where(atomic_numbers  == 1)[0] #array of 2
     O_pos = np.setdiff1d(ads_pos, H_pos)
     lens = []
     for k in np.arange(0, len(H_pos)):
        dist = atoms.get_distance(H_pos[k], O_pos[k], mic=True)
        lens.append(dist)
     #create dictionary entry
     results[sid[i]]= {'filenameOH': file[i], 'entry numberOH': entry[i], 
                    'compositionOH': atoms.get_chemical_formula(), 'adsorbate_OH_len': lens }
    
results_df = pd.DataFrame.from_dict(results, orient='index')  

breakpoint()
adE2 = adE.set_index('system_id_OH')
adE3 = results_df.join(adE2, how='inner')
adE3.to_csv('OER_angles_OH.csv')

#ex 1
dataset = LmdbDataset({'src': 'is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0000.lmdb'})
data = dataset[211] 
cell = data.cell.squeeze(0).numpy() 
atoms = Atoms( numbers=data.atomic_numbers.tolist(), positions=data.pos_relaxed.numpy(), cell=cell,  pbc=True)
tags = data.tags.numpy()
atomic_numbers = data.atomic_numbers.numpy()
ads_pos = np.where(tags==2)[0]
H_pos = np.where(atomic_numbers  == 1)[0] #array of 2
O_pos = np.setdiff1d(ads_pos, H_pos)
print(len(O_pos) == len(H_pos))
for k in np.arange(len(H_pos)):
     print(atoms.get_distance(H_pos[k], O_pos[k]))



