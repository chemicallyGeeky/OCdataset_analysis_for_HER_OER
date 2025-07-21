import pickle
import pandas as pd
from glob import iglob
from fairchem.core.datasets import LmdbDataset


def combine_lmdbs_and_metadata(lmdb_folder, meta_filename, output_filename):

    data_dict = {}
    for filename in iglob(lmdb_folder + "/data.lmdb"):
        for d in LmdbDataset({'src': filename}):
            data_dict[d['sid']] = {'y_relaxed': d['y_relaxed'], 'natoms': d['natoms']}

    data = pd.DataFrame(data_dict).T
    data = data.astype({'system_id': int, 'natoms': int})

    meta_data = pickle.load(open(meta_filename, 'rb'))
    md_df = data.apply(lambda x: pd.Series(meta_data[f"random{x.name}"]), axis=1)
    data = pd.concat([data, md_df], axis=1)
    data = data.rename(columns={"y_relaxed": "adsorption_energy"})
    data.to_csv('output_filename', index=True)
