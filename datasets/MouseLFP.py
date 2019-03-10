from timeit import default_timer as timer
from datasets.LFPDataset import LFPDataset
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "/data2/gabir/DATASETS_AS_NPYs/M017_S001_SRCS3L_25,50,100_0002.npy"


class MouseLFP(LFPDataset):
    def __init__(self, train_perc=0.60, val_perc=0.20, test_perc=0.20, random_seed=42, nr_bins=256):
        super().__init__(DATASET_PATH)

        np.random.seed(42)
        self.random_seed = random_seed
        self.nr_channels = len(self.channels)
        self.nr_bins = nr_bins

        self.lfp_data = {1: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                         2: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                         3: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length))}
        cur_element = [0, 0, 0, 0]

        """Iterate over all channels and place them in the dictionary. A channel has length = trial_length"""
        for i in range(0, self.channels.shape[1] // self.trial_length):
            condition_number = self.stimulus_conditions[i]
            self.lfp_data[condition_number][cur_element[condition_number], :, :] = self.channels[:,
                                                                                   (i * self.trial_length):((
                                                                                                                    i * self.trial_length) + self.trial_length)]
            cur_element[condition_number] += 1
        min_val = min(np.min(self.lfp_data[1]), np.min(self.lfp_data[2]), np.min(self.lfp_data[3]))
        max_val = max(np.max(self.lfp_data[1]), np.max(self.lfp_data[2]), np.max(self.lfp_data[3]))
        self.values_range = min_val, max_val
        self._get_train_val_test_split(train_perc, test_perc, val_perc)
        self._pre_compute_bins()

    def _pre_compute_bins(self):
        self.classes = {}
        min_train_seq = np.floor(self.values_range[0])
        max_train_seq = np.ceil(self.values_range[1])
        self.bins = np.linspace(min_train_seq, max_train_seq, self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]
        print("Pre computing classes for the dataset")
        start = timer()
        for movie in self.lfp_data.values():
            for value in movie.flatten():
                self.classes[value] = self._encode_input_to_bin(value)
        end = timer()
        print("Time needed for pre computing classes", end - start)

    def _encode_input_to_bin(self, target_val):
        bin = np.searchsorted(self.bins, target_val, side='left')
        return bin

    def _get_train_val_test_split(self, train_perc, val_perc, test_perc, random=False):
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
            self.test = {1: self.lfp_data[1][:, :,
                            self.train_length + self.val_length:self.train_length + self.val_length + self.test_length],
                         2: self.lfp_data[2][:, :,
                            self.train_length + self.val_length:self.train_length + self.val_length + self.test_length],
                         3: self.lfp_data[3][:, :,
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
            y.append(self.classes[next_step_value] if classifying else next_step_value)
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
            y.append(self.classes[next_step_value] if classifying else next_step_value)
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
            temp = self.test[movie_index][trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(self.classes[temp] if classifying else temp)
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
        return self.lfp_data[movie][trial, channel, :]

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
    dataset = MouseLFP()
    print(dataset.channels)