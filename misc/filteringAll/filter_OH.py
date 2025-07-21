import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import ast

filename = 'angles_OH.csv' 
adE3 = pd.read_csv(filename)
adE3['OH lengths'] = adE3['OH lengths'].apply(ast.literal_eval)
filtered = adE3.copy()

lowOH = 0.9; highOH = 1.1
filtered = adE3[adE3['OH lengths'].apply(lambda x: all(lowOH <= val <= highOH for val in x))]

#keep max value and plot
adE4 = adE3.copy()
adE4['maxOH'] = adE4['OH lengths'].apply(max)
fig, ax = plt.subplots()
ax.scatter(adE4['maxOH'], adE3['adsorption_energy'], color='k', s=1)
ax.axvspan(xmin=lowOH, xmax= highOH, color='g', alpha=0.3)
plt.xlabel('OH bond length', fontsize=24, fontweight='bold')
plt.ylabel('Adsorption energy/ eV', fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
