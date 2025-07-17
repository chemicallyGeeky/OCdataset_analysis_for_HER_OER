import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import ast
plt.rcParams['mathtext.default'] = 'regular'

filename = 'OER_angles_OH.csv' 
adE3 = pd.read_csv(filename, index_col=0) #sid of OH
adE3['adsorbate OH len'] = adE3['adsorbate OH len'].apply(ast.literal_eval)
filtered = adE3.copy() #128
print(filtered.keys())

lowOH = 0.9; highOH = 1.1
filtered = adE3[adE3['adsorbate OH len'].apply(lambda x: all(lowOH <= val <= highOH for val in x))] #128 to 92

#keep max value and plot
adE4 = adE3.copy()
adE4['maxOH'] = adE4['adsorbate OH len'].apply(max)
fig, ax = plt.subplots()
ax.scatter(adE4['maxOH'], adE4['eta'], color='k', s=1)
ax.axvspan(xmin=lowOH, xmax= highOH, color='g', alpha=0.3)
plt.xlabel('OH bond length /$\mathrm{\AA}$', fontsize=24, fontweight='bold')
plt.ylabel(r'$\eta_{TD}$ /V', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
plt.show()

filtered.to_csv('OERfiltered-OOH-OH.csv')

filtered2 = filtered[['bulk_id', 'miller_index', 'bulk_symbols', 'eta']]
filtered2 = filtered2.sort_values(by='eta')

good = filtered2[filtered2['eta']< 0.5]
print(good)
good.to_excel('goodOER_filtered.xlsx')
