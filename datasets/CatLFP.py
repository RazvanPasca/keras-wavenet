# from timeit import default_timer as timer
from datasets.LFPDataset import LFPDataset
from datasets.DATASET_PATHS import CAT_DATASET_PATH
import matplotlib.pyplot as plt
import numpy as np


class CatLFP(LFPDataset):
    def __init__(self, val_perc=0.20, test_perc=0.20, random_seed=42, nr_bins=256, nr_of_seqs=3):
        super().__init__(CAT_DATASET_PATH)
        self.nr_bins = nr_bins

        self._compute_values_range()
        self._pre_compute_bins()
        self._split_lfp_into_movies()
        self._get_train_val_test_split(test_perc, val_perc)

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.prediction_sequences = {
            'val': [self.get_random_sequence_from('VAL') for _ in range(nr_of_seqs)],
            'train': [self.get_random_sequence_from('TRAIN') for _ in range(nr_of_seqs)]
        }

    def _split_lfp_into_movies(self):
        self.all_lfp_data = []
        for condition_id in range(0, self.number_of_conditions):
            condition = []
            for i in range(0, self.channels.shape[1] // self.trial_length):
                if self.stimulus_conditions[i] == condition_id + 1:
                    trial = self.channels[:, (i * self.trial_length):((i * self.trial_length) + self.trial_length)]
                    condition.append(trial)
            self.all_lfp_data.append(np.array(condition))
        self.all_lfp_data = np.array(self.all_lfp_data)
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
            self.train = self.all_lfp_data[:, :, :, :self.train_length]
            self.validation = self.all_lfp_data[:, :, :, self.train_length:self.train_length + self.val_length]
            self.test = self.all_lfp_data[:, :, :,
                        self.train_length + self.val_length:self.train_length + self.val_length + self.test_length]

    def frame_generator(self, frame_size, batch_size, classifying, data):
        x = []
        y = []
        while 1:
            random_sequence, _ = self.get_random_sequence(data, frame_size + 1)
            frame = random_sequence[:-1]
            next_step_value = random_sequence[-1]
            x.append(frame.reshape(frame_size, 1))
            y.append(self._encode_input_to_bin(next_step_value) if classifying else next_step_value)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def train_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.train)

    def validation_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.validation)

    def test_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.test)

    def get_dataset_piece(self, movie, trial, channel):
        return self.all_lfp_data[movie][trial, channel, :]

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

    def get_random_sequence_from(self, source='VAL'):
        if source == 'TRAIN':
            data, data_addr = self.get_random_sequence(self.train, self.train.shape[-1])
        elif source == 'VAL':
            data, data_addr = self.get_random_sequence(self.validation, self.validation.shape[-1])
        elif source == 'TEST':
            data, data_addr = self.get_random_sequence(self.test, self.test.shape[-1])
        data_addr['SOURCE'] = source
        return data, data_addr

    def get_random_sequence(self, data, seq_length):
        batch_start = np.random.choice(range(0, data.shape[-1] - seq_length + 1))
        data_address = {
            'M': batch_start % self.number_of_conditions,
            'T': batch_start % self.trials_per_condition,
            'C': batch_start % self.nr_channels
        }
        return data[data_address['M'], data_address['T'], data_address['C'],
               batch_start:batch_start + seq_length], data_address


if __name__ == '__main__':
    dataset = CatLFP()
    print(dataset.all_lfp_data.shape)
