import pandas as pd
import ast
import numpy as np

def get_oh_filter(lowOH=0.9, highOH=1.1):
    oh_columns = ['OH lengths']
    oh_geo = pd.read_csv("data/oc2022/OH_geometry.csv", index_col="sid",
                         converters={c: ast.literal_eval for c in oh_columns})

    return oh_geo["OH lengths"].apply(lambda lengths: all(lowOH <= length <= highOH for length in lengths))

def get_ooh_filter(lowOH=0.9, highOH=1.1,
            lowOO=1.3, highOO=1.5,
            lowOOH=95, highOOH=115):

    ooh_columns = ['OH length 1', 'OH length 2', 'OO length', 'OOH angle 1', 'OOH angle 2']


    ooh_geo = pd.read_csv("data/oc2022/OOH_geometry.csv", index_col="sid",
                          converters={c: ast.literal_eval for c in ooh_columns})


    def filter_OOH(row):

        row = row.apply(np.array)

        cond = ((
            ((lowOO <= row['OO length']) & (row['OO length'] <= highOO))
            & ((lowOOH <= row['OOH angle 1']) & (row['OOH angle 1'] <= highOOH))
            & ((lowOH <= row['OH length 1']) & (row['OH length 1'] <= highOH)))
            | (((lowOO <= row['OO length']) & (row['OO length'] <= highOO))
               & ((lowOOH <= row['OOH angle 2']) & (row['OOH angle 2'] <= highOOH))
               & ((lowOH <= row['OH length 2']) & (row['OH length 2'] <= highOH))))

        return all(cond)

    return ooh_geo.apply(filter_OOH, axis=1)

def remove_bad_adsorbates(oer_data, lowOH=0.9, highOH=1.1,
                          lowOO=1.3, highOO=1.5,
                          lowOOH=95, highOOH=115):

    oh_filter = get_oh_filter(lowOH, highOH)

    ooh_filter = get_ooh_filter(lowOH, highOH, lowOO, highOO, lowOOH, highOOH)
    
    return oer_data[oh_filter.loc[oer_data["system_id_OH"]].to_numpy() & ooh_filter.loc[oer_data["system_id_HO2"]].to_numpy()].copy()
