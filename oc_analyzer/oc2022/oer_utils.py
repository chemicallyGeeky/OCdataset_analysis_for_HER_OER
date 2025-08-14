import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def gibbs_correction(row, correction_dict):
    """
    Applies Gibbs free energy correction based on the adsorbate symbol.
    Assumes adsorption energy *per adsorbate*.
    """
    return correction_dict.get(row["ads_symbols"], 0) + row["adsorption_energy"]

def scaling(x_vals, y_vals):
    reg = LinearRegression()
    reg.fit(x_vals.reshape(-1, 1), y_vals)
    return reg.intercept_, reg.coef_[0]

def merge_slabs_with_all_adsorbates(df, ads_list, miller=False):

    if miller:
        matching_columns = ['slab_sid', "miller_index"]
    else:
        matching_columns = 'slab_sid'

    merged_df = df[df['ads_symbols'] == ads_list[0]].copy()
    for ads in ads_list[1:]:
        ads_df = df[df['ads_symbols'] == ads]
        merged_df = pd.merge(merged_df, ads_df, how='inner',
                             left_on=matching_columns, right_on=matching_columns,
                             suffixes=('', f'_{ads}'))

    duplicated_cols = [name.strip("_" + ads_list[-1])
                       for name in merged_df.columns
                       if "_" + ads_list[-1] in name]

    merged_df = merged_df.rename(columns={col: col + "_" + ads_list[0] for col in duplicated_cols})

    merged_df["bulk_symbols"] = merged_df["bulk_symbols_" + ads_list[0]]
    
    return merged_df

def compute_oer_eta(df):
    df = compute_oer_reaction_energies(df)
    df['maxG'] = df[['delG1', 'delG2', 'delG3', 'delG4']].apply(max, axis=1)
    df['RDS'] = df.apply(get_rds, axis=1)
    df['eta'] = df['maxG'] - 1.23
    return df

def compute_oer_reaction_energies(df):
    df['delG1'] = df['adsorption_free_energy_OH']
    df['delG2'] = df['adsorption_free_energy_O'] - df['adsorption_free_energy_OH']
    df['delG3'] = df['adsorption_free_energy_HO2'] - df['adsorption_free_energy_O']
    df['delG4'] = 4.92 - df['adsorption_free_energy_HO2']
    return df

def get_rds(row):
    rn1 = row['delG1']
    rn2 = row['delG2']
    rn3 = row['delG3']
    rn4 = row['delG4']
    maxVal = np.argmax(np.array([rn1, rn2, rn3, rn4]))
    return maxVal + 1
