import pandas as pd
import matplotlib.pyplot as plt
from oc_analyzer.oc2022.oer_utils import gibbs_correction, merge_slabs_with_all_adsorbates, compute_oer_eta
import oc_analyzer.plot_utils as pltu

def main():

    data = pd.read_csv('data/oc2022/adsorption_energies.csv', index_col=0, na_values='')

    correction_dict = {"OH": 0.26, "O": -0.03, "HO2": 0.22}

    data['adsorption_free_energy'] = data.apply(gibbs_correction, args=(correction_dict,), axis=1)

    oer_data = merge_slabs_with_all_adsorbates(data, ('OH', 'O', 'HO2'))

    oer_data = compute_oer_eta(oer_data)

    oer_data["eta_approx"] = oer_data["delG2"] - 1.23

    fig, ax_main_left, ax_top, ax_right = pltu.create_main_panels(ae_limits=(-5, 5),
                                                                  eta_limits=(5, 0),
                                                                  xlabel=r'${\Delta G_{O} - \Delta G_{OH}} - 1.23$')


    uncertainty = 0.7 + 0.4
    
    pltu.add_shadded_regions(ax_main_left, ax_top, ax_right, uncertainty=uncertainty)

    pltu.plot_main_panel(ax_main_left, ax_top, ax_right, oer_data,
                         xlabel='eta_approx',
                         lit=None, special_samples=None, s=10, alpha=0.2)

    pltu.plot_distributions(ax_top, ax_right, oer_data,
                            xlabel='eta_approx', uncertainty=uncertainty)

    plt.savefig("oer.svg")

    print("Percentage of catalysts within uncertainty: ",
          len(oer_data[(oer_data['eta'] < uncertainty)]) / len(oer_data) * 100)

    plt.show()

if __name__ == "__main__":
    main()
