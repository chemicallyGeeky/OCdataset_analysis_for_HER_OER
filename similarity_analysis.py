import pandas as pd
import numpy as np
from oc_analyzer.oc2022.filter import get_oh_filter, get_ooh_filter
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os

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
    dists_to_mean = []
    max_diffs = []
    for i, dat in enumerate(repeated_pairs.iterrows()):
        index, row = dat
        condition = np.all((duplicated_df[subset] == row[subset]).to_numpy(), axis=1)
        same_pair = duplicated_df[condition]

        if 'nads' in same_pair.columns and 'slab_sid' in same_pair.columns:
            same_slab = same_pair.set_index(['slab_sid','nads', 'ads_symbols'])
        else:
            same_slab = same_pair.set_index(['ads_symbols'])
            
        tmp_dict = {}
        def instance_number(x):
            y = tmp_dict[x.name] = tmp_dict.get(x.name,0) + 1
            return y

        same_slab["extra_index"] = same_slab.apply(instance_number, axis=1)

        same_slab = same_slab.set_index("extra_index", append=True).unstack(level=-2) # So that the second instance of an adsorbate is on a different line

        same_slab = same_slab.loc[:, same_slab.notna().sum(axis=0) > 1] # Remove singles
        
        if same_slab.empty:
            continue

        if set(df["ads_symbols"].unique()) == set(['*OH', '*O', '*OOH']):
            adsorbate_order = ['*OH', '*O', '*OOH']
        else:
            adsorbate_order = df["ads_symbols"].unique()
        
        dist_to_mean = same_slab[quantity] - same_slab[quantity].mean()

        dist_to_mean = dist_to_mean.reindex(columns=adsorbate_order)

        max_diff = same_slab[quantity].max() - same_slab[quantity].min()

        max_diff = max_diff.reindex(index=adsorbate_order)
        
        max_diffs.append(same_slab[quantity].max() - same_slab[quantity].min())
        
        df_list.append(same_pair)
        dists_to_mean.append(dist_to_mean)

    n_groups = pd.DataFrame([df.notna().any() for df in dists_to_mean]).sum()
    dists_to_mean = pd.concat(dists_to_mean)

    print("Covariance matrix:")
    cov_matrix = dists_to_mean.cov()
    print(cov_matrix)

    print("Sizes:")
    sizes = dists_to_mean.notna().map(int).T@dists_to_mean.notna().map(int)
    print(sizes)

    mask = dists_to_mean.notna().map(int).to_numpy()
    
    super_mask = np.expand_dims(mask,2)@np.expand_dims(mask,1)

    dtm_array = dists_to_mean.to_numpy()

    dtm_array[np.isnan(dtm_array)] = 0
    
    cov_vars = ((super_mask*np.expand_dims(dtm_array,2))**2).sum(axis=0)/(super_mask.sum(axis=0)-1)

    print("Pearson correlation matrix:")
    pearson_mat = cov_matrix / (np.sqrt(cov_vars)*np.sqrt(cov_vars.T))
    print(pearson_mat)
    
    g = sns.pairplot(dists_to_mean, aspect=1, diag_kind='kde')

    for ax in g.axes.flatten():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        # Add light gray grid
        ax.grid(True, color='lightgray', linewidth=0.5, alpha=0.7)
        # Add slightly darker lines at x=0 and y=0
        ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.8)
        ax.axvline(x=0, color='gray', linewidth=0.8, alpha=0.8)

    # Save the plot with dynamic filename from dataframe
    adsorbates_str = "_".join(adsorbate_order)
    subset_str = "_".join(subset)
    plot_name = f"pairplot_{adsorbates_str}_{subset_str}".replace(" ", "_").replace("*", "")
    g.savefig(f"paper/figures/{plot_name}.svg", dpi=300, bbox_inches='tight')
    g.savefig(f"paper/figures/{plot_name}.pdf", dpi=300, bbox_inches='tight')

    plt.show()

    main_sizes = pd.Series(np.diag(sizes),sizes.columns)

    sigmas = pd.Series(np.diag(np.sqrt(cov_matrix)),cov_matrix.columns)

    props = pd.concat([n_groups, main_sizes],axis=1).apply(lambda x: f"{(x[0])} / {x[1]}", axis=1)
    
    out_df = pd.DataFrame({'$n_{\\text{groups}}$ / $n_{\\text{struc.}}$': props, 
                           "STD$^*$ (eV)": sigmas,
                           "Max diff (eV)": pd.DataFrame(max_diffs).max(axis=0)})

    all_df = pd.DataFrame({'$n_{\\text{groups}}$ / $n_{\\text{struc.}}$':
                           f"{sum(n_groups)} / {sum(main_sizes)}",
                           "STD$^*$ (eV)": np.nanstd(dists_to_mean.to_numpy()),
                           "Max diff (eV)": pd.DataFrame(max_diffs).max(axis=None)}, index=["All"])

    all_df.index.name = "ads_symbols"

    out_df = pd.concat([out_df, all_df])
    
    return {"Cov matrix": cov_matrix,
            "Pearson matrix": pearson_mat,
            "Sizes": sizes,
            "df": out_df}

def print_stats(data, adsorbates, oc22=True):
    data_dict = {}
    print(f"{' and '.join(adsorbates)} adsorption energies:")
    print("################################################################")
    print("Same miller (termination + coverage + site)")
    print("--------------------------------------------------------------")
    data_dict["Same miller (termination + coverage + site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['bulk_id', 'miller_index'], quantity='adsorption_energy')
    print("--------------------------------------------------------------")
    if oc22:
        print("Same miller and nads (termination + site)")
        print("--------------------------------------------------------------")
        data_dict["Same miller and $n_\\text{ads}$ (termination + site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['bulk_id', 'miller_index','nads'], quantity='adsorption_energy')
        print()
        print("Same slab ID (coverage + site)")
        print("--------------------------------------------------------------")
        data_dict["Same slab ID (coverage + site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['slab_sid'], quantity='adsorption_energy')
        print("--------------------------------------------------------------")
        print()
        print("Same slab_sid and nads (site)")
        print("--------------------------------------------------------------")
        data_dict["Same slab ID and $n_\\text{ads}$ (site)"] = group_similar(data[data['ads_symbols'].isin(adsorbates)], subset=['slab_sid', 'nads'], quantity='adsorption_energy')
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

    # Rename adsorbates from the start
    oc22_data = oc22_data.replace({'ads_symbols': {'OH': '*OH', 'O': '*O', 'HO2': '*OOH'}})

    print("OC22 data:")
    oc22_dict = print_stats(oc22_data, adsorbates=['*OH', '*O', '*OOH'])
    oc22_basic_stats = pd.concat({k:v["df"] for k,v in oc22_dict.items()}).stack().unstack("ads_symbols")
    
    # Clean up the dataframe
    oc22_basic_stats.columns.name = None
    oc22_basic_stats = oc22_basic_stats[['*OH', '*O', '*OOH', 'All']]

    print()

    pd.DataFrame(oc22_basic_stats).to_latex("paper/tables/oc22_similarity_stats.tex", float_format=lambda x: f"{x:.2f}")
    
    oc22_advanced_stats = pd.concat({k:pd.concat({kk:v[kk] for kk in ["Cov matrix", "Pearson matrix", "Sizes"]}, axis=1) for k,v in oc22_dict.items()})
    oc22_advanced_stats.index.names = [None, None]
    oc22_advanced_stats.columns.names = [None, None]
    
    pd.DataFrame(oc22_advanced_stats).to_latex("paper/tables/oc22_advanced_similarity_stats.tex", float_format=lambda x: f"{x:.2f}")
    
    # Save advanced stats as YAML
    advanced_stats_yaml = {}
    for scenario, scenario_data in oc22_dict.items():
        advanced_stats_yaml[scenario] = {
            "Cov matrix": scenario_data["Cov matrix"].to_dict(),
            "Pearson matrix": scenario_data["Pearson matrix"].to_dict(),
        }
    
    with open("data/oc2022/advanced_stats.yaml", "w") as f:
        yaml.dump(advanced_stats_yaml, f, default_flow_style=False)

    print("OC20 data:")
    oc20_dict = print_stats(oc20_data, adsorbates=['*H'],oc22=False)
    oc20_basic_stats = pd.concat({k:v["df"] for k,v in oc20_dict.items()}).stack().unstack("ads_symbols")   
    
    # Clean up the dataframe
    oc20_basic_stats.columns.name = None
    oc20_basic_stats = oc20_basic_stats[['*H']]
    
    pd.DataFrame(oc20_basic_stats).to_latex("paper/tables/oc20_similarity_stats.tex", float_format=lambda x: f"{x:.2f}")
