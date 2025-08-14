import pandas as pd
import matplotlib.pyplot as plt

from oc_analyzer.oc2022.oer_utils import merge_slabs_with_all_adsorbates
from oer_figure import remove_bad_adsorbates
import oc_analyzer.plot_utils as pltu

def main():
    data = pd.read_csv('data/oc2022/adsorption_energies.csv', na_values='')
    stability_data = pd.read_csv('data/oc2022/stability.csv', na_values='')

    oer_data = merge_slabs_with_all_adsorbates(data, ('OH', 'O', 'HO2'), miller=True)

    tolerance_dict = {"lowOH": 0.9, "highOH": 1.1,
                      "lowOO": 1.3, "highOO": 1.5,
                      "lowOOH": 95, "highOOH": 115}

    oer_data_filtered = remove_bad_adsorbates(oer_data, **tolerance_dict)

    uncertainty = 0.1

    stability_data["entry_id"] = stability_data["entry_id"].apply(lambda x: "-".join(x.split("-")[:2]))
    stability = stability_data[stability_data["voltage"] == 1.63]
    stability = stability[stability["decomposition_energy"] > -1e-5] # exclude negatives
    stability_filter = stability["entry_id"].isin(oer_data_filtered["bulk_id_OH"])

    pltu.plot_stability_distribution(stability, stability_filter, uncertainty)

    print("Average decomposition energy:", stability['decomposition_energy'].mean())
    print("Average decomposition energy (filtered):", stability.loc[stability_filter, 'decomposition_energy'].mean())

    print("Percentage of stable structures within uncertainty:",
          (stability['decomposition_energy'] < uncertainty).sum() / len(stability) * 100,
          (stability['decomposition_energy'] < uncertainty).sum())


    print("Percentage of stable filtered structures within uncertainty:",
          (stability.loc[stability_filter, "decomposition_energy"] < uncertainty).sum() / sum(stability_filter) * 100,
          (stability.loc[stability_filter, "decomposition_energy"] < uncertainty).sum())


    plt.savefig("paper/figures/stability.svg")
    plt.savefig("paper/figures/stability.pdf")

    plt.show()

if __name__ == "__main__":
    main()
