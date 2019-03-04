import json
import os

import matplotlib.pyplot as plt
import numpy as np


class LFPDataset:
    def __init__(self, data_path, random_seed=42):
        np.random.seed(42)
        with open(data_path, 'r+') as f:
            lfp_data = json.loads(f.read())
        with open(os.path.join(os.path.dirname(data_path), lfp_data['stimulus_condition_file']), 'r+') as f:
            lfp_data['stimulusOrder'] = [int(st) for st in f.read().split('\n')]
        lfp_data['channels'] = np.array(
            [np.fromfile(open(os.path.join(os.path.dirname(data_path), file), 'rb'), np.float32) for file in
             lfp_data['bin_file_names']])

        self.file_path = data_path
        self.random_seed = random_seed
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
        self.nr_channels = len(self.channels)
        self.lfp_data = {1: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                         2: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                         3: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length))}
        cur_element = [0, 0, 0, 0]

        """Iterate over all channels and place them in the dictionary. A channel has length = trial_length"""
        for i in range(0, self.channels.shape[1] // self.trial_length):
            condition_number = self.stimulusOrder[i]
            self.lfp_data[condition_number][cur_element[condition_number], :, :] = self.channels[:,
                                                                                   (i * self.trial_length):((
                                                                                                                    i * self.trial_length) + self.trial_length)]
            cur_element[condition_number] += 1
        self.trial_length = lfp_data['trial_length']
        min_val = min(np.min(self.lfp_data[1]), np.min(self.lfp_data[2]), np.min(self.lfp_data[3]))
        max_val = max(np.max(self.lfp_data[1]), np.max(self.lfp_data[2]), np.max(self.lfp_data[3]))
        self.values_range = min_val, max_val
        self._get_train_val_test_split()

    def _get_train_val_test_split(self, train_perc=0.60, val_perc=0.20, test_perc=0.20, random=False):
        self.train_length = round(train_perc * self.trial_length)
        self.val_length = round(val_perc * self.trial_length)
        self.test_length = round(test_perc * self.trial_length)
        if not random:
            self.train = {1: self.lfp_data[1][:, :, :self.train_length],
                          2: self.lfp_data[2][:, :, :self.train_length],
                          3: self.lfp_data[3][:, :, :self.train_length]}
            self.validation = {1: self.lfp_data[1][:, :, self.train_length:self.train_length + self.val_length],
                               2: self.lfp_data[2][:, :, self.train_length:self.train_length + self.val_length],
                               3: self.lfp_data[3][:, :, self.train_length:self.train_length + self.val_length]}
            self.test = {1: self.lfp_data[1][:, :, self.train_length + self.val_length:],
                         2: self.lfp_data[2][:, :, self.train_length + self.val_length:],
                         3: self.lfp_data[3][:, :, self.train_length + self.val_length:]}

    def train_frame_generator(self, frame_size, batch_size, output_transform):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, self.train_length - frame_size - 1))
            movie_index = batch_start % self.number_of_conditions + 1
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = self.train[movie_index][trial_index, channel_index, batch_start:batch_start + frame_size]
            temp = self.train[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(output_transform(temp))
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def validation_frame_generator(self, frame_size, batch_size, output_transform):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, self.val_length - frame_size - 1))
            movie_index = batch_start % self.number_of_conditions + 1
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = self.validation[movie_index][trial_index, channel_index, batch_start:batch_start + frame_size]
            temp = self.validation[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(output_transform(temp))
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
            x = []
            y = []

    def test_frame_generator(self, frame_size, batch_size, output_transform):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, self.test_length - frame_size - 1))
            movie_index = batch_start % self.number_of_conditions + 1
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = self.test[movie_index][trial_index, channel_index, batch_start:batch_start + frame_size]
            temp = self.test[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(output_transform(temp))
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
            x = []
            y = []

    def plot_signal(self, movie, trial, channel, start=0, stop=None, save=False):
        if stop is None:
            stop = self.trial_length
        plt.figure(figsize=(16, 12))
        plot_title = "Movie:{}_Channel:{}_Trial:{}_Start:{}_Stop:{}".format(movie, channel, trial, start, stop)
        plt.title(plot_title)
        plt.plot(self.get_dataset_piece(movie, trial, channel)[start:stop], label="LFP signal")
        plt.legend()
        if save:
            plt.savefig(
                "/home/pasca/School/Licenta/Datasets/CER01A50/Plots/" + "{}/".format(movie) + plot_title + ".png")
        # plt.show()

    def get_dataset_piece(self, movie, trial, channel):
        return self.lfp_data[movie][trial, channel, :]

    def get_total_length(self, partition):
        if partition == "TRAIN":
            return self.train[1].size + self.train[2].size + self.train[3].size
        elif partition == "VAL":
            return self.validation[1].size + self.validation[2].size + self.validation[3].size
        elif partition == "TEST":
            return self.test[1].size + self.test[2].size + self.test[3].size
        else:
            raise ValueError("Please pick a valid partition from: TRAIN, VAL and TRAIN")
