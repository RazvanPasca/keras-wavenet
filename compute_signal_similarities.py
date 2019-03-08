from itertools import product

import numpy as np

from LFP_Dataset import LFPDataset

dataset = LFPDataset("/home/razpa/CER01A50/Bin_cer01a50-LFP.json")


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


movies = range(1, 4)
trials = range(dataset.trials_per_condition)
channels = range(dataset.nr_channels)

combinations = list(product(*[movies, trials, channels]))

pair_sim_list = []

for movie1, trial1, channel1 in combinations[:-1]:
    for movie2, trial2, channel2 in combinations[1:]:
        signal1 = dataset.get_dataset_piece(movie1, trial1, channel1)
        signal2 = dataset.get_dataset_piece(movie2, trial2, channel2)
        sim = rmse(signal1, signal2)
        s1 = "M:{}_T:{}_C:{}".format(movie1, trial1, channel1)
        s2 = "M:{}_T:{}_C:{}".format(movie2, trial2, channel2)
        pair_sim_list.append((s1 + "|" + s2, sim))

pair_sim_list.sort(key=lambda x: x[1])

with open("similarities.txt", "a+") as file:
    for pair in pair_sim_list:
        file.write(pair[0] + ": " + str(pair[1]) + '\n')
