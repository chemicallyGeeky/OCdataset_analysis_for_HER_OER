#run from folder with pickle file
#source catalyst/bin/activate
import pickle 
import numpy as np
import pandas as pd

mdata = pickle.load(open("oc20_data_mapping.pkl", "rb")) #loads a dict
#The keys of this dictionary are the system-ids in the lmdb datasets
print ('dataset length = ', len(mdata))
system_id = list(mdata.keys())

# The dict has the following entries: 'bulk_id', 'ads_id', 'bulk_mpid', 'bulk_symbols', 'ads_symbols', 
# 'miller_index', 'shift', 'top', 'adsorption_site', 'class', 'anomaly'
s = {}
for i in system_id:
    try:
        s[i] = pd.concat([pd.Series({'system_id':i}),pd.Series(mdata[i])])
    except KeyError:
        s[i] = pd.Series({'system_id': i,'bulk_mpid': np.NaN, 'miller_index': np.NaN, 
                          'bulk_symbols': np.NaN, 'ads_symbols': np.NaN, 'class': np.NaN,
                            'anomaly': np.NaN})  
#s is now a dict with a series as its value for each key
#CHECK that s[i] and mdata[i] match

column_names = ['system_id','bulk_mpid', 'miller_index', 'bulk_symbols', 'ads_symbols', 'class', 'anomaly']
cols = {}
for j in range(len(column_names)):
        cols[j] = []
        for i in system_id: 
            try:
                cols[j].append(s[i][column_names[j]])
            except KeyError:
                cols[j].append(np.NaN)


#create dataframe
df0 = pd.DataFrame([cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6]], 
                  index=['system_id','bulk_mpid', 'miller_index', 'bulk_symbols', 'ads_symbols', 'class', 'anomaly' ]).T

#clean 
df = df0.set_index('system_id')
df = df.sort_index()
#CHECK: Now, mdata[i], s[i] and df.loc[i] should match

#remove 'random' from system_id
df.index = [x.replace('random', '') for x in df.index]
df.index = df.index.astype('int')

#this may not be required
df = df.astype({'bulk_mpid': 'str', 'bulk_symbols':'str', 'miller_index': 'str', 'ads_symbols':'str'})
names = ['bulk_mpid', 'miller_index', 'bulk_symbols', 'ads_symbols']
for name in names:
    df[name] = [x.strip() for x in df[name]]

#explore
print('unique compositions = ', df0['bulk_symbols'].nunique(), 'unique surfaces = ', df0['miller_index'].nunique())
print(' unique mp ids = ', df0['bulk_mpid'].nunique())

#export
df.to_csv('fromMetadataFinal.csv')

