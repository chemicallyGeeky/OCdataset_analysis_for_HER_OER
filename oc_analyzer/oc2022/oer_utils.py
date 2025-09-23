import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from tqdm import tqdm
from scipy.interpolate import CubicSpline

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

def softmax(x, k):
    return np.sum(np.exp(k*x) * x / np.sum(np.exp(k*x)))

def softmax_grad(x, k, uncertainty):
    dg = np.array([uncertainty, 2*uncertainty, 2*uncertainty, uncertainty])
    return np.sum(((k*x*np.exp(k*x)+np.exp(k*x))/(np.sum(np.exp(k*x))) - np.sum(np.exp(k*x)*x)*k*np.exp(k*x)/(np.sum(np.exp(k*x)))**2)*dg)

def softmax_torch(row, k, uncertainty):
    dg = torch.tensor(row.loc[['delG1', 'delG2', 'delG3', 'delG4']].array, requires_grad=True)
    sm = torch.nn.Softmax(dim=0)
    gmax = torch.sum(sm(k*dg) * dg)
    gmax.backward()
    uncertainty_prop = torch.sum(dg.grad * torch.tensor([uncertainty, 2 * uncertainty, 2 * uncertainty, uncertainty]))
    row["softmaxG_torch"] = gmax.item()
    row["uncertainty_grad_torch"] = uncertainty_prop.item()
    return row

def bootstrap_gmax(uncertainty, n_samples=1000, grid_size=100, n_batch=3):
    gmax_mean = [[[None]*n_batch for _ in range(n_batch)] for _ in range(n_batch)]
    gmax_std = [[[None]*n_batch for _ in range(n_batch)] for _ in range(n_batch)]
    lin = np.linspace(0, 4.92, grid_size)
    lin_splits = np.split(lin, n_batch)
    for i in tqdm(range(n_batch)):
        for j in tqdm(range(n_batch)):
            for k in tqdm(range(n_batch)):
                e_oh = np.random.normal(0, uncertainty, size=(grid_size//n_batch, grid_size//n_batch, grid_size//n_batch, n_samples)) + lin_splits[i].reshape(-1, 1, 1, 1)
                e_o = np.random.normal(0, uncertainty, size=(grid_size//n_batch, grid_size//n_batch, grid_size//n_batch, n_samples)) + lin_splits[j].reshape(1, -1, 1, 1)
                e_ho2 = np.random.normal(0, uncertainty, size=(grid_size//n_batch, grid_size//n_batch, grid_size//n_batch, n_samples)) + lin_splits[k].reshape(1, 1, -1, 1)
    
                gmax = np.max(np.array([e_oh, e_o - e_oh, e_ho2 - e_o, 4.92 - e_ho2]), axis=0)

                gmax_mean[i][j][k] = np.mean(gmax, axis=3)
                gmax_std[i][j][k] = np.std(gmax, axis=3)
                
    return np.block(gmax_mean), np.block(gmax_std)

def unbiased_bootstrap(row, gmax_grid, std_grid):
    
    argmin = np.unravel_index(np.argmin(abs(gmax_grid - row['maxG'])), gmax_grid.shape)
    
    row["maxG_unbiased_bootstrap"] = gmax_grid[argmin]
    row["uncertainty_unbiased_bootstrap"] = std_grid[argmin]

    return row
    
def uncertainty_bootstrap(row, uncertainty, n_samples=1000):

    E_oh = row['adsorption_free_energy_OH'] + np.random.normal(0, uncertainty, size=n_samples)
    E_o = row['adsorption_free_energy_O'] + np.random.normal(0, uncertainty, size=n_samples)
    E_ho2 = row['adsorption_free_energy_HO2'] + np.random.normal(0, uncertainty, size=n_samples)

    gmax = np.max(np.array([E_oh, E_o - E_oh, E_ho2 - E_o, 4.92 - E_ho2]), axis=0)
    
    row["softmaxG_bootstrap"] = np.mean(gmax)
    row["uncertainty_bootstrap"] = np.std(gmax)
    return row

def get_ideal_distr_OER(uncetainty = 0.3, n_surfaces = 1000000, output_size = 1000, best_known=0):
    
    energies = np.random.normal(1.23, uncetainty, size=(n_surfaces, 3))
    
    energies[:,1] += 1.23 + best_known
    energies[:,2] += 2*1.23
    
    energies = np.concatenate((np.zeros((n_surfaces, 1)), energies, np.ones((n_surfaces, 1))*1.23*4), axis=1)
    
    dgs = energies[:,1:] - energies[:,:-1]
    
    etas = dgs.max(axis=1) - 1.23
    
    print(f"Mean eta: {np.mean(etas)}, Std eta: {np.std(etas)}, Min eta: {np.min(etas)}, Max eta: {np.max(etas)}")

    mean_eta = np.mean(etas)
    std_eta = np.std(etas)

    pdf, eta = np.histogram(etas, bins=1000, density=True)
    eta = (eta[1:] + eta[:-1])/2
    cdf = np.cumsum(pdf)*(eta[1] - eta[0])

    # Interpolate to get a smooth curve
    pdf_cs = CubicSpline(eta, pdf, bc_type='natural', extrapolate=True)
    cdf_cs = CubicSpline(eta, cdf, bc_type='natural', extrapolate=True)
    
    def cdf(x):
        out = cdf_cs(x)
        out[x <= 0] = 0.0
        out[x >= eta[-1]] = 1.0
        return out

    def pdf(x):
        out = pdf_cs(x)
        out[x <= 0] = 0.0
        out[x >= eta[-1]] = 0.0
        return out
        
    return etas, pdf, cdf

def get_ideal_distr_HER(uncetainty = 0.3, n_surfaces = 1000000, output_size = 1000):

    energies = np.random.normal(0, uncetainty, size=(n_surfaces, 3))

    etas = abs(energies)

    pdf, eta = np.histogram(etas, bins=1000, density=True)
    eta = (eta[1:] + eta[:-1])/2
    cdf = np.cumsum(pdf)*(eta[1] - eta[0])

    # Interpolate to get a smooth curve
    pdf = CubicSpline(eta, pdf, bc_type='natural', extrapolate=True)
    cdf_cs = CubicSpline(eta, cdf, bc_type='natural', extrapolate=True)

    def cdf(x):
        out = cdf_cs(x)
        out[x <= 0] = 0.0
        out[x >= eta[-1]] = 1.0
        return out
    
    return eta, pdf, cdf
    
def uncertainty_propagation(data, uncertainty, k=10):
    data.loc[:, "softmaxG"] = data.loc[:, ['delG1', 'delG2', 'delG3', 'delG4']].apply(softmax, args=(k,), axis=1)
    data.loc[:, "uncertainty"] = data.loc[:, 'RDS'].apply(lambda x: 2*uncertainty if x in [2,3] else uncertainty)
    data.loc[:, "uncertainty_grad"] = data.loc[:, ['delG1', 'delG2', 'delG3', 'delG4']].apply(softmax_grad, args=(k, uncertainty), axis=1)
    data = data.apply(softmax_torch, axis=1, args=(k, uncertainty))
    data = data.apply(uncertainty_bootstrap, axis=1, args=(uncertainty, 1000))

    gmax_grid, std_grid = bootstrap_gmax(0.5, n_samples=1000, grid_size=99, n_batch=3)

    breakpoint()
    
    data = data.apply(unbiased_bootstrap, axis=1, args=(gmax_grid, std_grid))

    print(data[["maxG", "softmaxG_torch", "softmaxG_bootstrap", "maxG_unbiased_bootstrap", "uncertainty_grad_torch", "uncertainty_bootstrap", "uncertainty_unbiased_bootstrap"]].mean(axis=0))
    
    return data

def print_stats(data, uncertainty, best_known=0, treshold=0.6827):

    qualifier = "best known" if best_known > 0 else "ideal"
    
    ideal_etas, pdf, cdf = get_ideal_distr_OER(uncertainty, best_known=best_known)

    p = np.mean(1 - cdf(data['eta']))
    p_ideal = np.mean(1 - cdf(ideal_etas))
    print(f"Average probability of eta being at least as bad as what was sampled under the {qualifier} distribution: {p} ({p_ideal})")
    
    f = np.sum(cdf(data['eta']) < treshold)/len(data)
    f_ideal = np.sum(cdf(ideal_etas) < treshold)/len(ideal_etas)

    eta_tresh = ideal_etas[np.argmin(abs(cdf(ideal_etas) - treshold))]
    
    print(f"Percentage of samples with a p-value under the {qualifier} distribution of less than {100*(1-treshold)}%: {f} ({f_ideal})")
    
    f = np.sum(data['eta'] < ideal_etas.mean() + ideal_etas.std())/len(data)
    f_ideal = np.sum(ideal_etas < ideal_etas.mean() + ideal_etas.std())/len(ideal_etas)
    
    print(f"Percentage of samples within mean + std: {f} ({f_ideal})")
    print(f"Mean and STD of {qualifier} distribution: {ideal_etas.mean()} {ideal_etas.std()}")

    return pdf, eta_tresh
