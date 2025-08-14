import pandas as pd
import matplotlib.pyplot as plt
from oc_analyzer.oc2022.oer_utils import gibbs_correction, merge_slabs_with_all_adsorbates, compute_oer_eta
import oc_analyzer.plot_utils as pltu
import ast
import numpy as np


def remove_bad_adsorbates(oer_data, lowOH=0.9, highOH=1.1,
                          lowOO=1.3, highOO=1.5,
                          lowOOH=95, highOOH=115):

    oh_columns = ['OH lengths']
    ooh_columns = ['OH_length 1', 'OH_length 2', 'OO_length', 'OOH angle 1', 'OOH angle 2']

    oh_geo = pd.read_csv("data/oc2022/OH_geometry.csv", index_col="sid",
                         converters={c: ast.literal_eval for c in oh_columns})
    ooh_geo = pd.read_csv("data/oc2022/OOH_geometry.csv", index_col="sid",
                          converters={c: ast.literal_eval for c in ooh_columns})

    oh_filter = oh_geo["OH lengths"].apply(lambda lengths: all(lowOH <= length <= highOH for length in lengths))

    def filter_OOH(row):
    
        row = row.apply(np.array)

        cond = ((
            ((lowOO <= row['OO_length']) & (row['OO_length'] <= highOO))
            & ((lowOOH <= row['OOH angle 1']) & (row['OOH angle 1'] <= highOOH))
            & ((lowOH <= row['OH_length 1']) & (row['OH_length 1'] <= highOH)))
            | (((lowOO <= row['OO_length']) & (row['OO_length'] <= highOO))
               & ((lowOOH <= row['OOH angle 2']) & (row['OOH angle 2'] <= highOOH))
               & ((lowOH <= row['OH_length 2']) & (row['OH_length 2'] <= highOH))))

        return all(cond)

    ooh_filter = ooh_geo.apply(filter_OOH, axis=1)
    
    return oer_data[oh_filter.loc[oer_data["system_id_OH"]].to_numpy() & ooh_filter.loc[oer_data["system_id_HO2"]].to_numpy()]

def main():

    data = pd.read_csv('data/oc2022/adsorption_energies.csv', na_values='')

    correction_dict = {"OH": 0.26, "O": -0.03, "HO2": 0.22}  # From OC2022 paper

    data['adsorption_free_energy'] = data.apply(gibbs_correction, args=(correction_dict,), axis=1)

    oer_data = merge_slabs_with_all_adsorbates(data, ('OH', 'O', 'HO2'), miller=True)

    oer_data = compute_oer_eta(oer_data)

    oer_stability_data = oer_data.loc[:, ["bulk_id_OH", "eta"]]
    oer_stability_data.to_csv("oer_stability_data.csv", index=False)

    oer_data["eta_approx"] = oer_data["delG2"] - 1.23

    fig, ax_main_left, ax_top, ax_right = pltu.create_main_panels(ae_limits=(-5, 5),
                                                                  eta_limits=(5, 0),
                                                                  xlabel=r'${\Delta G_{O} - \Delta G_{OH}} - 1.23$')


    uncertainty = 0.7 + 0.4

    unfiltered_color = "gray"

    pltu.add_best_value(ax_main_left, ax_top, ax_right, best_val=0.4, color="lime")
    
    pltu.add_shadded_regions(ax_main_left, ax_top, ax_right, uncertainty=uncertainty)

    pltu.plot_main_panel(ax_main_left, ax_top, ax_right, oer_data,
                         xlabel='eta_approx',
                         lit=None, special_samples=None, s=10, alpha=0.2, color="gray",
                         label='OC22 DFT predictions')

    bins = pltu.plot_distributions(ax_top, ax_right, oer_data,
                                   xlabel='eta_approx', uncertainty=uncertainty, color="gray")

    print(f"Number of catalysts: {len(oer_data)} surfaces, {len(oer_data["bulk_id_OH"].unique())} materials")

    print("Percentage of catalysts within uncertainty: ",
          len(oer_data[(oer_data['eta'] < uncertainty)]) / len(oer_data) * 100)

    print("Percentage of catalysts outside 4.92 V: ",
          len(oer_data[(oer_data['eta'] > 4.92)]) / len(oer_data) * 100)

    # Now we can filter the data based on the adsorbate geometries
    print("Filtering based on adsorbate geometries...")

    filtered_color = "cornflowerblue"
    
    tolerance_dict = {"lowOH": 0.9, "highOH": 1.1,
                      "lowOO": 1.3, "highOO": 1.5,
                      "lowOOH": 95, "highOOH": 115}
   
    oer_data_filtered = remove_bad_adsorbates(oer_data, **tolerance_dict)

    special_samples = {
        'Ir1O3': {"description": r'IrO$_3$', "color": 'red', "manual_adjustment": 0},  # Close to good
        'Ru1Pt1O4': {"description": r'RuPtO$_4$', "color": 'red', "manual_adjustment": 0},  # Close to good
    }
    
    pltu.plot_main_panel(ax_main_left, ax_top, ax_right, oer_data_filtered,
                         xlabel='eta_approx',
                         lit=None, special_samples=special_samples, s=10,
                         alpha=1, color=filtered_color,
                         label='Filtered predictions')

    pltu.plot_distributions(ax_top, ax_right, oer_data_filtered,
                            xlabel='eta_approx', uncertainty=uncertainty, bins=bins, color=filtered_color)

    plt.savefig("paper/figures/oer.svg")
    plt.savefig("paper/figures/oer.pdf")

    print(f"Number of catalysts: {len(oer_data_filtered)} surfaces, {len(oer_data_filtered['bulk_id_OH'].unique())} materials")

    print("Percentage of catalysts within uncertainty: ",
          len(oer_data_filtered[(oer_data_filtered['eta'] < uncertainty)]) / len(oer_data_filtered) * 100)

    print("Percentage of catalysts outside 4.92 V: ",
          len(oer_data_filtered[(oer_data_filtered['eta'] > 4.92)]) / len(oer_data_filtered) * 100)

    plt.show()


if __name__ == "__main__":
    main()
