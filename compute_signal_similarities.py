import numpy as np

from LFP_Dataset import LFPDataset

dataset = LFPDataset("/home/pasca/School/Licenta/Datasets/CER01A50/Bin_cer01a50-LFP.json")


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


for movie in dataset.lfp_data.values():
    for trial in range(dataset.trials_per_condition):
        for channel in range(dataset.nr_channels - 1):
            sim = rmse(dataset.get_dataset_piece(movie, trial, channel))
