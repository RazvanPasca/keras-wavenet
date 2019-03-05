from itertools import product

import numpy as np

from LFP_Dataset import LFPDataset

dataset = LFPDataset("/home/pasca/School/Licenta/Datasets/CER01A50/Bin_cer01a50-LFP.json")


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


movies = range(1, 4)
trials = range(dataset.trials_per_condition)
channels = range(dataset.nr_channels)

combinations = list(product(*[movies, trials, channels]))

for movie1, trial1, channel1 in combinations[:-1]:
    for movie2, trial2, channel2 in combinations[1:]:
        signal1 = dataset.get_dataset_piece(movie1, trial1, channel1)
        signal2 = dataset.get_dataset_piece(movie2, trial2, channel2)
        sim = rmse(signal1, signal2)
        with open("similarities.txt", "w+") as file:
            file.write("Similarity between {} and {}:{}".format(str(movie1) + "_" + str(trial1) + "_" + str(channel1),
                                                                str(movie2) + "_" + str(trial2) + "_" + str(channel2),
                                                                sim))
