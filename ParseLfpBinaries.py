import numpy as np
import matplotlib.pyplot as plt
import json
import os


class LfpData:

    def __init__(self, file_path):
        with open(file_path, 'r+') as f:
            lfp_data = json.loads(f.read())
        with open(os.path.join(os.path.dirname(file_path), lfp_data['stimulus_condition_file']), 'r+') as f:
            lfp_data['stimulusOrder'] = [int(st) for st in f.read().split('\n')]
        lfp_data['channels'] = np.array(
            [np.fromfile(open(os.path.join(os.path.dirname(file_path), file), 'rb'), np.float32) for file in
             lfp_data['bin_file_names']])
        self.file_path = file_path
        self.bin_file_names = lfp_data['bin_file_names']
        self.trial_length = lfp_data['trial_length']
        self.sampling_frequency = lfp_data['sampling_frequency']
        self.number_of_lfp_files = lfp_data['number_of_lfp_files']
        self.ldf_file_version = lfp_data['ldf_file_version']
        self.stimulus_on_at = lfp_data['stimulus_on_at']
        self.stimulus_off_at = lfp_data['stimulus_off_at']
        self.stimulus_condition_file = lfp_data['stimulus_condition_file']
        self.number_of_conditions = lfp_data['number_of_conditions']
        self.trials_per_condition = lfp_data['trials_per_condition']
        self.stimulusOrder = lfp_data['stimulusOrder']
        self.channels = lfp_data['channels']


def ParseLfps(location):
    lfp_data = LfpData(location)

    # print(lfp_data)
    # print(lfp_data.channels.shape)

    movies = {1: np.zeros((20, 47, 28000)),
              2: np.zeros((20, 47, 28000)),
              3: np.zeros((20, 47, 28000))}
    curElement = [0, 0, 0, 0]
    for i in range(0, lfp_data.channels.shape[1] // 28000):
        conditionNumber = lfp_data.stimulusOrder[i]
        movies[conditionNumber][curElement[conditionNumber], :, :] = lfp_data.channels[:,
                                                                     (i * 28000):((i * 28000) + 28000)]
        curElement[conditionNumber] += 1

    # print(movies[1][0][:28000].shape)
    # plt.plot(movies[1][0][0][:1000])
    # plt.show()
    return movies

    # for i in movies:
    #     with open('data_python_dict_dump_movie' + str(i), 'w+') as f:
    #         f.write(movies[i].tobytes())
    #
    # for key in movies:
    #     print(movies[key].shape)

    # first_trial = lfp_data.channels[:, 28000*3:28000*3+50]
    # x = np.linspace(0, first_trial.shape[1], first_trial.shape[1])
    # for channel in first_trial:
    #     plt.plot(x, channel)
    # plt.show()
    #
    # first_trial = movies[1][1, :, :50]
    # x = np.linspace(0, first_trial.shape[1], first_trial.shape[1])
    # for channel in first_trial:
    #     plt.plot(x, channel)
    # plt.show()
