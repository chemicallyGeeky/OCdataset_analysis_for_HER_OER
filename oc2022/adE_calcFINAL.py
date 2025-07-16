#run from oc2022: calculate adsorption energy and compare methods
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

element_energies = {"H":-3.386, "O":-7.459, "C":-7.332, "N":-8.309}
allE = pd.read_csv('lmdbPlusMetaData.csv', index_col=0, na_values='')

#adsorbates
print('unique adsorbates = ', allE['ads_symbols'].nunique())
print('unique adsorbates = ', allE['ads_symbols'].unique())

#surfaces
print('surfaces = ', allE['miller_index'].nunique())
allSurfaces = list(allE['miller_index'].unique())

idx1 = []; idx2 = []; idx3 = []
for s in allSurfaces:
    idx1.append((re.findall('\((\d).*', s))[0])
    idx2.append((re.findall('\s(.*),', s))[0])
    idx3.append((re.findall('.*(\d)\)', s))[0])
millerIdx = pd.DataFrame([idx1, idx2, idx3], index=['idx1', 'idx2', 'idx3']).T    
millerIdx.to_csv('surfaces_oc22.csv')

#compositions
print('unique compositions = ', allE['bulk_symbols'].nunique())

def get_formula_dict(formula):
    formula_dict = {}
    for elem, n in re.findall("([A-Z][a-z]?)([0-9]*)", formula):
        if n == "":
            formula_dict[elem] = 1
        else:
            formula_dict[elem] = int(n)
    return formula_dict

print(get_formula_dict('H2SO4'), get_formula_dict('SrTiO3'))

print('Any 0 values in slab_sid? ', np.sum(allE['slab_sid']==0))
print('Any 0 values in index? ', np.sum(allE.index==0))
allE = allE.astype({'slab_sid':'Int64'})
breakpoint() #1
allE['slab_sid'] = allE['slab_sid'].fillna(0)

#method1
adsorption_E1 =[]    
for i in allE.index:
    s = allE.loc[i, 'slab_sid']
    n = allE.loc[i, 'nads']
    if (s == 0) or (n == 0):
        adsorption_E1.append(np.nan) #for pristine surface
    else:
        try:
            surface_E = allE.loc[s, 'y_relaxed']
            ads = allE.loc[i, 'ads_symbols']
            ads_formula = get_formula_dict(ads) #splits into atom symbol and number
            ads_E = np.sum([element_energies[k]*v for k,v in ads_formula.items()])
            adsorption_E1.append((allE.loc[i, 'y_relaxed'] - surface_E - n*ads_E )/n)
        #matching pristine surface cannot be found
        except KeyError:
            adsorption_E1.append(np.nan)

breakpoint() #2
adsorption_E1 = np.array(adsorption_E1)

#method 2

def get_ads_energy(row):
    n = row['nads']
    #pristine surface
    if (row["slab_sid"] == 0) or (n == 0) :
          return np.nan
    try:
        slab_E = allE.loc[int(row["slab_sid"]), "y_relaxed"]
    #matching pristine surface cannot be found
    except KeyError:
        return np.nan    
    ads_formula = get_formula_dict(row['ads_symbols'])
    ads_E = np.sum([element_energies[k]*v for k,v in ads_formula.items()])
    return (row['y_relaxed'] - slab_E - n*ads_E)/n

# allE["adsorption_energy"]
allE["adsorption_energy"] = allE.apply(get_ads_energy, axis=1)

#compare1
print('COMPARE')
print('Do the two methods give similar results? ', np.array_equal(np.array(allE["adsorption_energy"]), adsorption_E1, equal_nan=True))
print(np.nanmean(adsorption_E1), allE["adsorption_energy"].mean())
breakpoint() #3

#compare2
# Ir1O3 = allE[allE['bulk_symbols'] == 'Ir1O3']
# Ir1O3.to_csv('Ir1O3.csv')

#explore
print('EXPLORE')
print('average adsorption energy = ', allE['adsorption_energy'].mean())
maxIdx = allE['adsorption_energy'].idxmax()
minIdx = allE['adsorption_energy'].idxmin()
print('max. adsorption energy = ', allE['adsorption_energy'].max(), ' at ', maxIdx)
print('min. adsorption energy = ', allE['adsorption_energy'].min(), ' at ', minIdx)
print(allE.loc[maxIdx], allE.loc[minIdx])

#sort
#sortedByAdE = allE.reset_index()
allE.to_csv('allAdE.csv')
sortedByAdE = allE.dropna(subset=['adsorption_energy']) #remove pristine surfaces etc. w.o. ad E
sortedByAdE2 = sortedByAdE.sort_values(by='adsorption_energy') #arrange in order of adsorption energy
#drop outliers
sortedByAdE2 = sortedByAdE2.iloc[1:len(sortedByAdE)-4,:]
#plot distribution
plt.figure; plt.hist(sortedByAdE2['adsorption_energy'])
plt.title('Adsorption Energy (ev) Distribution'); plt.savefig('hist')

#export
sortedByAdE3 = sortedByAdE2.sort_values(by='bulk_symbols')
sortedByAdE3.to_csv('sortedWithAdE.csv')
print('number of adosrption energies = ', len(sortedByAdE3))

breakpoint() #4

