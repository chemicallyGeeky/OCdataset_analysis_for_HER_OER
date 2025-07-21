#filtering with OOH bond lengths and angles

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['mathtext.default'] = 'regular'

filename = 'OERangles_OOH.csv' 
adE3 = pd.read_csv(filename, index_col=0)
filtered = adE3.copy()
filtered = filtered.sort_values('eta')
filtered = filtered[~filtered.index.duplicated(keep='first')] #keep duplicate with lower eta value
#491 - 412

#ranges
lowOO = 1.3; highOO = 1.5
lowOOH = 95; highOOH = 115
lowOH = 0.9; highOH = 1.1

#adsorbate 1
cond1 = (
    (filtered['OO_length'].between(lowOO, highOO)) & 
    (filtered['OOH angle 1'].between(lowOOH, highOOH)) & 
    (filtered['OH_length 1'].between(lowOH, highOH))
)  
cond2 = (
    (filtered['OO_length'].between(lowOO, highOO)) & 
    (filtered['OOH angle 2'].between(lowOOH, highOOH)) & 
    (filtered['OH_length 2'].between(lowOH, highOH))
)

filtered = filtered[cond1 | cond2] #412-128

#adsorbate 2
cond1b = (
        (filtered['OO_lengthb'].between(lowOO, highOO) | filtered['OO_lengthb'].isna()) &
        (filtered['OOH angle 1b'].between(lowOOH, highOOH) | filtered['OOH angle 1b'].isna()) &
        (filtered['OH_length 1b'].between(lowOH, highOH) | filtered['OH_length 1b'].isna())
        )
cond2b = (
        (filtered['OO_lengthb'].between(lowOO, highOO) | filtered['OO_lengthb'].isna()) & 
         (filtered['OOH angle 2b'].between(lowOOH, highOOH) | filtered['OOH angle 2b'].isna()) & 
         (filtered['OH_length 2b'].between(lowOH, highOH) | filtered['OH_length 2b'].isna())
         )
filtered = filtered[cond1b | cond2b] #128-128
#checked that none of the 3 systems with ads > 1 are there

# only keep the correct angles: only need to do this for adsrobate 1
filtered['OOHangle'] = np.where(
    filtered['OOH angle 1'].between(lowOOH, highOOH), filtered['OOH angle 1'],
    np.where(filtered['OOH angle 2'].between(lowOOH, highOOH), filtered['OOH angle 2'], np.nan))

filtered['OH_length'] = np.where(
    filtered['OH_length 1'].between(lowOH, highOH), filtered['OH_length 1'],
    np.where(filtered['OH_length 2'].between(lowOH, highOH), filtered['OH_length 2'], np.nan))

filtered2 = filtered[['composition', 'OO_length', 'miller_index', 'OOHangle', 'OH_length', 'eta']]
filtered.to_csv('oerFiltered_OOH.csv')

plt.scatter(filtered2['OOHangle'], filtered2['eta'], color='k', s=1)
plt.xlabel('OOH angle', fontsize=24, fontweight='bold')
plt.ylabel(r'$\eta_{TD}$ /V', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

plt.scatter(filtered2['OO_length'], filtered2['eta'], color='k', s=1)
plt.xlabel('OO bond length /($\mathrm{\AA}$)', fontsize=24, fontweight='bold')
plt.ylabel(r'$\eta_{TD}$ /V', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#filtered.to_csv('filtered.csv')

#PLOTS without filtering
fig, ax = plt.subplots()
ax.scatter(adE3['OO_length'], adE3['eta'], color='k', s=1)
ax.axvspan(xmin=lowOO, xmax= highOO, color='g', alpha=0.3)
plt.xlabel('OO bond length /$\mathrm{\AA}$', fontsize=24, fontweight='bold')
plt.ylabel(r'$\eta_{TD}$ /V', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
plt.show()

#either 1 or 2
fig, ax = plt.subplots()
ax.scatter(adE3['OOH angle 1'], adE3['eta'], color='k', s=1)
ax.axvspan(xmin=lowOOH, xmax= highOOH, color='g', alpha=0.3)
plt.xlabel('OOH angle /$^\circ$', fontsize=24, fontweight='bold')
plt.ylabel(r'$\eta_{TD}$ /V', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
plt.show()

#either 1 or 2
fig, ax = plt.subplots()
ax.scatter(adE3['OH_length 1'], adE3['eta'], color='k', s=1)
ax.axvspan(xmin=lowOH, xmax= highOH, color='g', alpha=0.3)
plt.xlabel('OH bond length /$\mathrm{\AA}$', fontsize=24, fontweight='bold')
plt.ylabel(r'$\eta_{TD}$ /V', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')
plt.show()


