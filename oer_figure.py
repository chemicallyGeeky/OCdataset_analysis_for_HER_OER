import pandas as pd
import matplotlib.pyplot as plt
from oc_analyzer.oc2022.oer_utils import gibbs_correction, merge_slabs_with_all_adsorbates, compute_oer_eta, uncertainty_propagation, get_ideal_distr_OER, print_stats
from oc_analyzer.oc2022.filter import remove_bad_adsorbates
import oc_analyzer.plot_utils as pltu
import numpy as np

def main():

    materials_only = False  # If True, only the best surface per material is kept
    surface_selection = "last"
    uncertainty = 0.5
    best_known = 0.4
    unfiltered_color = "gray"
    filtered_color = "cornflowerblue"
    tolerance_dict = {"lowOH": 0.9, "highOH": 1.1,
                      "lowOO": 1.3, "highOO": 1.5,
                      "lowOOH": 95, "highOOH": 115}
    # tolerance_dict = {"lowOH": 0.8, "highOH": 1.2,
    #                   "lowOO": 1.4, "highOO": 1.6,
    #                   "lowOOH": 80, "highOOH": 135}
    correction_dict = {"OH": 0.26, "O": -0.03, "HO2": 0.22}  # From OC2022 paper

    data = pd.read_csv('data/oc2022/adsorption_energies.csv', na_values='')
    
    data['adsorption_free_energy'] = data.apply(gibbs_correction, args=(correction_dict,), axis=1)

    oer_data = merge_slabs_with_all_adsorbates(data, ('OH', 'O', 'HO2'), miller=True)

    oer_data = compute_oer_eta(oer_data)

    oer_stability_data = oer_data.loc[:, ["bulk_id_OH", "eta"]]
    oer_stability_data.to_csv("oer_stability_data.csv", index=False)

    oer_data["eta_approx"] = oer_data["delG2"] - 1.23

    # Now we can filter the data based on the adsorbate geometries
    oer_data_filtered = remove_bad_adsorbates(oer_data, **tolerance_dict)

    if materials_only:
        oer_data = oer_data.sort_values("eta").drop_duplicates(subset=["bulk_id_OH"], keep=surface_selection)
        oer_data_filtered = oer_data_filtered.sort_values("eta").drop_duplicates(subset=["bulk_id_OH"], keep=surface_selection)
    
    print_stats(oer_data_filtered, uncertainty, best_known=0)

    pdf, eta_tresh = print_stats(oer_data_filtered, uncertainty, best_known=best_known)
    
    fig, ax_main_left, ax_top, ax_right = pltu.create_main_panels(ae_limits=(-5, 5),
                                                                  eta_limits=(5, 0),
                                                                  xlabel=r'${\Delta G_{*O} - \Delta G_{*OH}} - 1.23$')

    pltu.add_best_value(ax_main_left, ax_top, ax_right, best_val=best_known, color="lime")
    
    pltu.add_shadded_regions(ax_main_left, ax_top, ax_right, uncertainty=eta_tresh)

    pltu.plot_main_panel(ax_main_left, ax_top, ax_right, oer_data,
                         xlabel='eta_approx',
                         lit=None, special_samples=None, s=10, alpha=0.2, color="gray",
                         label='OC22 DFT predictions')

    bins = pltu.plot_distributions(ax_top, ax_right, oer_data,
                                   xlabel='eta_approx', uncertainty=eta_tresh, color="gray")

    print(f"Number of unfiltered catalysts: {len(oer_data)} surfaces, {len(oer_data["bulk_id_OH"].unique())} materials")

    print("Treashold for uncertainty: ", eta_tresh)

    print("Percentage of catalysts within treshold: ",
          len(oer_data[(oer_data['eta'] < eta_tresh)]) / len(oer_data) * 100)

    print("Percentage of catalysts outside 4.92 V: ",
          len(oer_data[(oer_data['eta'] > 4.92)]) / len(oer_data) * 100)
    
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
                            xlabel='eta_approx', uncertainty=eta_tresh, bins=bins, color=filtered_color, ideal_pdf=pdf)
    
    plt.savefig("paper/figures/oer.svg")
    plt.savefig("paper/figures/oer.pdf")

    print(f"Number of filtered catalysts: {len(oer_data_filtered)} surfaces, {len(oer_data_filtered['bulk_id_OH'].unique())} materials")

    print("Percentage of catalysts within treshold: ",
          len(oer_data_filtered[(oer_data_filtered['eta'] < eta_tresh)]) / len(oer_data_filtered) * 100)

    print("Percentage of catalysts outside 4.92 V: ",
          len(oer_data_filtered[(oer_data_filtered['eta'] > 4.92)]) / len(oer_data_filtered) * 100)

    plt.show()


if __name__ == "__main__":
    main()
