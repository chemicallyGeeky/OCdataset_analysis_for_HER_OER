import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle
import oc_analyzer.plot_utils as pltu
from fairchem.core.datasets import LmdbDataset

def combine_lmdbs_and_metadata(lmdb_folder, meta_filename, output_filename):

    data_dict = {}
    for filename in iglob(lmdb_folder+ "/data.lmdb"):
        for d in LmdbDataset({'src':filename}):
            data_dict[d['sid']] = {'y_relaxed': d['y_relaxed'], 'natoms': d['natoms']}
              
    data = pd.DataFrame(data_dict).T
    data = data.astype({'system_id': int, 'natoms': int})
    
    meta_data = pickle.load(open(meta_filename, 'rb'))
    md_df = data.apply(lambda x: pd.Series(meta_data[f"random{x.name}"]), axis=1)
    data = pd.concat([data, md_df], axis=1)
    data = data.rename(columns={"y_relaxed": "adsorption_energy"})
    data.to_csv('output_filename', index=True)
    

def get_danilov_data(comp_data, closest=True):
    # data from: https://doi.org/10.1002/anie.201204842
    lit = pd.read_csv('danilovic.csv', index_col="Catalyst")

    sorted_array = np.sort(lit.to_numpy())

    sorted_array[:,0] = sorted_array[:,1] - sorted_array[:,0]
    sorted_array[:,2] = sorted_array[:,2] - sorted_array[:,1]

    ordered_lit = pd.DataFrame(sorted_array, index=lit.index, columns=["-", "median", "+"])

    lit = pd.concat([lit, ordered_lit], axis=1)

    for catal in lit.index:
        same_catal_list = comp_data[comp_data['bulk_symbols'].apply(lambda x: "".join(re.findall("[A-Za-z]{1,2}", x)) == catal)]

        if closest:
            oc20_idx = abs(same_catal_list['eta'] - lit.loc[catal,"HClO4"]).idxmin()
        else:
            oc20_idx = same_catal_list['eta'].idxmin()

        lit.loc[catal,"database_value"] = comp_data.loc[oc20_idx, 'adsorption_free_energy']

    lit["experimental_value"] = lit["HClO4"]

    return lit


def main():
    # Uncomment the next line to combine LMDB data with metadata

    # combine_lmdbs_and_metadata("is2res_train_val_test_lmdbs/data/is2re/100k/train", "oc20_data_mapping.pkl", "processed_data/oc2020/lmdb+metadata.csv")
    
    oc20_data = pd.read_csv('processed_data/oc2020/lmdb+metadata.csv', index_col=0, na_values = '')

    her_data = oc20_data.copy()
    her_data = her_data[her_data['ads_symbols'] == '*H']

    her_data['adsorption_free_energy'] = her_data['adsorption_energy'] + 0.24
    her_data['eta'] = abs(her_data['adsorption_free_energy'])

    lit = get_danilov_data(her_data)

    special_samples = {
        'Ca': {"description":'Ca', "color": 'red', "manual_adjustment": -0.1},  # reacts with acidic conditions
        'Na': {"description": 'Na', "color": 'red', "manual_adjustment": -0.05},  # explodes in water
        'K4': {"description": 'K', "color": 'red', "manual_adjustment": 0.05},  # explodes in water
        'Pt': {"description": 'Pt', "color": 'lime', "manual_adjustment": 0}  # best known catalyst
    }

    fig, ax_main_left, ax_top, ax_right = pltu.create_main_panels(ae_limits=(-2, 2), eta_limits=(2, 0), xlabel=r'${\Delta G}_H$')

    pltu.add_shadded_regions(ax_main_left, ax_top, ax_right, uncertainty=0.3)

    pltu.plot_main_panel(ax_main_left, ax_top, ax_right, her_data, xlabel='adsorption_free_energy', lit=lit, special_samples=special_samples)

    pltu.plot_distributions(ax_top, ax_right, her_data, xlabel='adsorption_free_energy', uncertainty=0.3)

    #gs.tight_layout(fig, rect=[0, 0, 0.9, 0.9])  # Adjust layout to prevent overlap
    plt.savefig("her.svg")

    print("Percentage of catalysts within uncertainty: ", len(her_data[(her_data['eta'] < 0.3)]) / len(her_data) * 100)

    plt.show()


if __name__ == "__main__":
    main()
