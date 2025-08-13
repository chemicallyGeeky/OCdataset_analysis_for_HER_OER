import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.gridspec as gridspec

filename = 'data/oc2020/lmdb+metadata.csv'
adE = pd.read_csv(filename, index_col=0, na_values='')

maskH =  (adE['ads_symbols'] == '*H')  
adE_H = adE[maskH] 
mask = adE_H['anomaly'] == 0
adE_H = adE_H[mask]

adE_H = adE_H.sort_values(by=['bulk_mpid', 'miller_index'])

df = (adE_H.groupby(['bulk_mpid', 'miller_index', 'anomaly'])['adsorption_energy'].nunique().reset_index(name='count'))
duplicates = df[df['count']>1]

H_duplicates = adE_H.reset_index().merge(
    duplicates.drop(columns='count'),
    on=['bulk_mpid', 'miller_index', 'anomaly'],
    how='inner')

H_duplicates['diff'] = H_duplicates.groupby(['bulk_mpid', 'miller_index', 'anomaly'])['adsorption_energy'].diff().abs()

print('same bulk_mpid, miller_index')
print('min, mean, max: ', H_duplicates['diff'].min(), H_duplicates['diff'].mean(), H_duplicates['diff'].max())
#min, mean, max:  0.00037131999999928667 0.4088835325454581 1.568975660000035

print(len(H_duplicates))
print(len(H_duplicates[H_duplicates['anomaly']==0]))



