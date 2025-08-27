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

    rmse = np.sqrt(np.mean(np.concatenate(dists_to_mean)**2))
    # mae = np.mean(abs(np.concatenate(dists_to_median)))
    # mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)

    print("Average square distance with mean (RMSE like): ",
          np.sqrt(np.mean(np.concatenate(dists_to_mean)**2)))
    print("Average difference with median (MAE like): ",
          np.mean(abs(np.concatenate(dists_to_median))))
    print("Average difference between extremes: ", np.mean(diffs))
    print("Maximum difference between extremes: ", np.max(diffs))

    return {'$n_{\\text{groups}}$ / $n_{\\text{struc.}}$':
            f"{(len(repeated_pairs))} / {len(duplicated_df)}",
            "RMSE$^*$ (eV)": f"{rmse:.2f}",
            # "MAE$^*$)": mae,
            # "Mean diff": mean_diff,
            "Max diff (eV)": f"{max_diff:.2f}"}


def print_stats(data, adsorbates, oc22=True):
    data_dict = {}
    print(f"{' and '.join(adsorbates)} adsorption energies:")
    print("################################################################")
    print("Same miller (termination + coverage + site)")
    print("--------------------------------------------------------------")
    data_dict["Same miller (termination + coverage + site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['ads_symbols', 'bulk_id', 'miller_index'], quantity='adsorption_energy')
    print("--------------------------------------------------------------")
    if oc22:
        print("Same miller and nads (termination + site)")
        print("--------------------------------------------------------------")
        data_dict["Same miller and $n_\\text{ads}$ (termination + site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['ads_symbols', 'bulk_id', 'miller_index','nads'], quantity='adsorption_energy')
        print()
        print("Same slab ID (coverage + site)")
        print("--------------------------------------------------------------")
        data_dict["Same slab ID (coverage + site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['ads_symbols', 'slab_sid'], quantity='adsorption_energy')
        print("--------------------------------------------------------------")
        print()
        print("Same slab_sid and nads (site)")
        print("--------------------------------------------------------------")
        data_dict["Same slab ID and $n_\\text{ads}$ (site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['ads_symbols', 'slab_sid', 'nads'], quantity='adsorption_energy')
        print("--------------------------------------------------------------")
    print("################################################################")
    print()

    return data_dict


if __name__ == "__main__":
    # Load data
    filter_data = True
    oc22_data = pd.read_csv('data/oc2022/adsorption_energies.csv', index_col=0)
    oc20_data = pd.read_csv('data/oc2020/lmdb+metadata.csv', index_col=0)

    oc20_data = oc20_data.rename(columns={"bulk_mpid": "bulk_id"})

    if filter_data:
        print("Filtering out bad adsorbates:")
        oh_filter = get_oh_filter()
        ooh_filter = get_ooh_filter()
        oc22_data = oc22_data.drop(oh_filter[~oh_filter].index)
        oc22_data = oc22_data.drop(ooh_filter[~ooh_filter].index)
        oc20_data = oc20_data[oc20_data['anomaly'] == 0]

    print("OC22 data:")
    oc22_dict = {}
    oc22_dict["*OH"] = pd.DataFrame(print_stats(oc22_data, adsorbates=['OH'])).T.stack()
    oc22_dict["*O"] = pd.DataFrame(print_stats(oc22_data, adsorbates=['O'])).T.stack()
    oc22_dict["*OOH"] = pd.DataFrame(print_stats(oc22_data, adsorbates=['HO2'])).T.stack()
    oc22_dict["All"] = pd.DataFrame(print_stats(oc22_data, adsorbates=['OH', 'O', 'HO2'])).T.stack()
    print()

    pd.DataFrame(oc22_dict).to_latex("paper/tables/oc22_similarity_stats.tex")

    print("OC20 data:")
    oc20_dict = {}
    oc20_dict["*H"] = pd.DataFrame(print_stats(oc20_data, adsorbates=['*H'],oc22=False)).T.stack()

    pd.DataFrame(oc20_dict).to_latex("paper/tables/oc20_similarity_stats.tex")
