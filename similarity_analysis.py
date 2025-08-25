import pandas as pd
import numpy as np
from oc_analyzer.oc2022.filter import get_oh_filter, get_ooh_filter

def group_similar(df, subset, quantity):
    # Check duplicates
    n_data = len(df)
    all_dup_rows = df.duplicated(subset=subset, keep=False)
    dup_rows = df.duplicated(subset=subset, keep="first")
    duplicated_df = df[all_dup_rows]
    unique_pair_df = df[~dup_rows]

    print("Number of unique data points: ", len(unique_pair_df), "/", n_data)

    repeated_pairs = duplicated_df.drop_duplicates(subset=subset)

    print("Number of repeated pairings: ", len(repeated_pairs), "/", len(duplicated_df))

    df_list = []
    diffs = []
    dists_to_mean = []
    dists_to_median = []
    for i, dat in enumerate(repeated_pairs.iterrows()):
        index, row = dat
        condition = np.all((duplicated_df[subset] == row[subset]).to_numpy(), axis=1)
        same_pair = duplicated_df[condition]
        df_list.append(same_pair)
        diffs.append(same_pair[quantity].max() - same_pair[quantity].min())
        dists_to_mean.append(same_pair[quantity] - same_pair[quantity].mean())
        dists_to_median.append(same_pair[quantity] - same_pair[quantity].mean())
    
    print("Average square distance with mean (RMSE like): ", np.sqrt(np.mean(np.concatenate(dists_to_mean)**2)))
    print("Average difference with median (MAE like): ", np.mean(abs(np.concatenate(dists_to_median))))
    print("Average difference between extremes: ", np.mean(diffs))
    print("Maximum difference between extremes: ", np.max(diffs))

def print_stats(data, adsorbate, oc22=True):
    print(f"{adsorbate} adsorption energies:")
    print("################################################################")
    print("Same miller (termination + coverage + site)")
    print("--------------------------------------------------------------")
    group_similar(data[data['ads_symbols'] == adsorbate], subset=['bulk_id', 'miller_index'], quantity='adsorption_energy')
    print("--------------------------------------------------------------")
    if oc22:
        print()
        print("Same slab_sid (coverage + site)")
        print("--------------------------------------------------------------")
        group_similar(data[data['ads_symbols'] == adsorbate], subset=['slab_sid'], quantity='adsorption_energy')
        print("--------------------------------------------------------------")
        print()
        print("Same slab_sid and nads (site)")
        print("--------------------------------------------------------------")
        group_similar(data[data['ads_symbols'] == adsorbate], subset=['slab_sid', 'nads'], quantity='adsorption_energy')
        print("--------------------------------------------------------------")
    print("################################################################")
    print()
    

if __name__ == "__main__":
    # Load data
    oc22_data = pd.read_csv('data/oc2022/adsorption_energies.csv', index_col=0)
    oc20_data = pd.read_csv('data/oc2020/lmdb+metadata.csv', index_col=0)

    oc20_data = oc20_data.rename(columns={"bulk_mpid": "bulk_id"})

    print("OC22 data:")
    print_stats(oc22_data, adsorbate='OH')
    print_stats(oc22_data, adsorbate='O')
    print_stats(oc22_data, adsorbate='HO2')
    print()
    print("OC20 data:")
    print_stats(oc20_data, adsorbate='*H',oc22=False)
    print()
    print()
    
    print("After filtering out bad adsorbates:")
    oh_filter = get_oh_filter()
    ooh_filter = get_ooh_filter()
    oc22_data = oc22_data.drop(oh_filter[~oh_filter].index)
    oc22_data = oc22_data.drop(ooh_filter[~ooh_filter].index)
    oc20_data = oc20_data[oc20_data['anomaly'] == 0]

    print("OC22 data:")
    print_stats(oc22_data, adsorbate='OH')
    print_stats(oc22_data, adsorbate='O')
    print_stats(oc22_data, adsorbate='HO2')
    print()
    print("OC20 data:")
    print_stats(oc20_data, adsorbate='*H',oc22=False)
