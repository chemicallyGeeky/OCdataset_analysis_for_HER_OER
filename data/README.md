# Data

- `danilovic.csv`: Overpotential Data extracted from: https://doi.org/10.1002/anie.201204842
- `norskov.csv`: Exchange current data extracted from: https://doi.org/10.1149/1.1856988
- `oc2020/lmdb+metadata.csv`: Lmbd energy information combined with metadata from the OC2020 dataset. This file can be obtained by running:
```
from oc_analyzer.oc2020 import combine_lmdbs_and_metadata
combine_lmdbs_and_metadata("is2res_train_val_test_lmdbs/data/is2re/100k/train",
                           "oc20_data_mapping.pkl",
                           "data/oc2020/lmdb+metadata.csv")
```
- `/oc2022`
