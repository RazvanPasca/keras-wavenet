# from timeit import default_timer as timer
from datasets.LFPDataset import LFPDataset
from datasets.DATASET_PATHS import CAT_DATASET_PATH
import matplotlib.pyplot as plt
import numpy as np


class CatLFP(LFPDataset):
    def __init__(self, val_perc=0.20, test_perc=0.20, random_seed=42, nr_bins=256):
        super().__init__(CAT_DATASET_PATH)
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.nr_bins = nr_bins

        self._compute_values_range()
        self._pre_compute_bins()
        self._split_lfp_into_movies()
        self._get_train_val_test_split(test_perc, val_perc)

    def _compute_values_range(self):
        min_val = np.min(self.channels)
        max_val = np.max(self.channels)
        self.values_range = min_val, max_val

    def _split_lfp_into_movies(self):
        self.all_lfp_data = {1: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                             2: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                             3: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length))}
        cur_element = [0, 0, 0, 0]
        """Iterate over all channels and place them in the dictionary. A channel has length = trial_length"""
        for i in range(0, self.channels.shape[1] // self.trial_length):
            condition_number = self.stimulus_conditions[i]
            self.all_lfp_data[condition_number][cur_element[condition_number], :, :] = self.channels[:,
                                                                                       (i * self.trial_length):((
                                                                                                                        i * self.trial_length) + self.trial_length)]
            cur_element[condition_number] += 1
        self.channels = None

    def _pre_compute_bins(self):
        self.cached_val_bin = {}
        min_train_seq = np.floor(self.values_range[0])
        max_train_seq = np.ceil(self.values_range[1])
        self.bins = np.linspace(min_train_seq, max_train_seq, self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]
        # print("Pre computing classes for the dataset")
        # start = timer()
        # for channel in self.channels:
        #     for v in channel:
        #         self.cached_val_bin[v] = self._encode_input_to_bin(v)
        # end = timer()
        # print("Time needed for pre computing classes", end - start)

    def _encode_input_to_bin(self, target_val):
        if target_val not in self.cached_val_bin:
            self.cached_val_bin[target_val] = np.digitize(target_val, self.bins, right=False)
        return self.cached_val_bin[target_val]

    def _get_train_val_test_split(self, val_perc, test_perc, random=False):
        self.val_length = round(val_perc * self.trial_length)
        self.test_length = round(test_perc * self.trial_length)
        self.train_length = self.trial_length - (self.val_length + self.test_length)
        if not random:
            self.train = {1: self.all_lfp_data[1][:, :, :self.train_length],
                          2: self.all_lfp_data[2][:, :, :self.train_length],
                          3: self.all_lfp_data[3][:, :, :self.train_length]}
            self.validation = {1: self.all_lfp_data[1][:, :, self.train_length:self.train_length + self.val_length],
                               2: self.all_lfp_data[2][:, :, self.train_length:self.train_length + self.val_length],
                               3: self.all_lfp_data[3][:, :, self.train_length:self.train_length + self.val_length]}
            self.test = {1: self.all_lfp_data[1][:, :,
                            self.train_length + self.val_length:self.train_length + self.val_length + self.test_length],
                         2: self.all_lfp_data[2][:, :,
                            self.train_length + self.val_length:self.train_length + self.val_length + self.test_length],
                         3: self.all_lfp_data[3][:, :,
                            self.train_length + self.val_length:self.train_length + self.val_length + self.test_length]}

    def train_frame_generator(self, frame_size, batch_size, classifying):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, self.train_length - frame_size - 1))
            movie_index = batch_start % self.number_of_conditions + 1
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = self.train[movie_index][trial_index, channel_index, batch_start:batch_start + frame_size]
            next_step_value = self.train[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(self._encode_input_to_bin(next_step_value) if classifying else next_step_value)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def validation_frame_generator(self, frame_size, batch_size, classifying):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, self.val_length - frame_size - 1))
            movie_index = batch_start % self.number_of_conditions + 1
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = self.validation[movie_index][trial_index, channel_index, batch_start:batch_start + frame_size]
            next_step_value = self.validation[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(self._encode_input_to_bin(next_step_value) if classifying else next_step_value)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def test_frame_generator(self, frame_size, batch_size, classifying):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, self.test_length - frame_size - 1))
            movie_index = batch_start % self.number_of_conditions + 1
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = self.test[movie_index][trial_index, channel_index, batch_start:batch_start + frame_size]
            next_step_value = self.test[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(self._encode_input_to_bin(next_step_value) if classifying else next_step_value)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def plot_signal(self, movie, trial, channel, start=0, stop=None, save=False, show=True):
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
        if show:
            plt.show()

    def get_dataset_piece(self, movie, trial, channel):
        return self.all_lfp_data[movie][trial, channel, :]

    def get_total_length(self, partition):
        if partition == "TRAIN":
            return self.train[1].size + self.train[2].size + self.train[3].size
        elif partition == "VAL":
            return self.validation[1].size + self.validation[2].size + self.validation[3].size
        elif partition == "TEST":
            return self.test[1].size + self.test[2].size + self.test[3].size
        else:
            raise ValueError("Please pick a valid partition from: TRAIN, VAL and TEST")


if __name__ == '__main__':
    dataset = CatLFP()
    print(dataset.channels)
    dataset.save_as_npy('./cer01a50.npy')
