from timeit import default_timer as timer
from datasets.LFPDataset import LFPDataset
from datasets.DATASET_PATHS import MOUSE_DATASET_PATH
import matplotlib.pyplot as plt
import numpy as np


class MouseLFP(LFPDataset):
    def __init__(self, val_perc=0.20, test_perc=0.20, random_seed=42, nr_bins=256, channels_to_consider=None):
        super().__init__(MOUSE_DATASET_PATH)
        np.random.seed(random_seed)
        self.channels = self.channels[:-1]  # Discard channel 33 which records heart beats
        self.nr_channels -= len(self.channels)
        if channels_to_consider is not None:
            self.channels = self.channels[np.array(channels_to_consider)]
        self.random_seed = random_seed
        self.nr_bins = nr_bins
        self.nr_of_orientations = 8
        self.nr_of_stimulus_luminosity_levels = 3
        self.trial_length = 2672  # 4175
        self._compute_values_range()
        self._pre_compute_bins()
        self._split_lfp_data()
        self._get_train_val_test_split(test_perc, val_perc)

    def _compute_values_range(self):
        min_val = np.min(self.channels)
        max_val = np.max(self.channels)
        self.values_range = min_val, max_val

    def _split_lfp_data(self):
        self.all_lfp_data = []
        for orientation in range(1, self.nr_of_orientations + 1):
            intensity = []
            for intensity_id in range(1, self.nr_of_stimulus_luminosity_levels + 1):
                condition = []
                condition_id = orientation * intensity_id
                for stimulus_condition in self.stimulus_conditions:
                    if stimulus_condition['Condition number'] == str(condition_id):
                        index = int(stimulus_condition['Trial'])
                        events = [{'timestamp': self.event_timestamps[4 * index + i],
                                   'code': self.event_codes[4 * index + i]} for i in range(4)]
                        trial = self.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)]
                        # Right now it cuts only the area where the stimulus is active
                        # In order to keep the whole trial replace with
                        # "trial = self.channels[:, events[0]['timestamp']:(events[0]['timestamp'] + 4175)]"
                        condition.append(trial)
                condition = np.array(condition)
                intensity.append(condition)
            self.all_lfp_data.append(intensity)
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
            self.train = self.all_lfp_data[:, :, :, :, :self.train_length]
            self.validation = self.all_lfp_data[:, :, :, :, self.train_length:self.train_length + self.val_length]
            self.test = self.all_lfp_data[:, :, :, :,
                        self.train_length + self.val_length:self.train_length + self.val_length + self.test_length]

    def frame_generator(self, frame_size, batch_size, classifying, length, data):
        x = []
        y = []
        while 1:
            batch_start = np.random.choice(range(0, length - (frame_size + 1)))
            orientation_index = batch_start % self.nr_of_orientations
            luminosity_index = batch_start % self.nr_of_stimulus_luminosity_levels
            trial_index = batch_start % self.trials_per_condition
            channel_index = batch_start % self.nr_channels
            frame = data[orientation_index, luminosity_index, trial_index, channel_index,
                    batch_start:batch_start + frame_size]
            next_step_value = data[
                orientation_index, luminosity_index, trial_index, channel_index, batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(self._encode_input_to_bin(next_step_value) if classifying else next_step_value)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def train_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.train_length, self.train)

    def validation_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.val_length, self.validation)

    def test_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.test_length, self.test)

    def get_dataset_piece(self, orientation, luminosity, trial, channel):
        return self.all_lfp_data[orientation, luminosity, trial, channel, :]

    def plot_signal(self, orientation, luminosity, trial, channel, start=0, stop=None, save=False, show=True):
        if stop is None:
            stop = self.trial_length
        plt.figure(figsize=(16, 12))
        plot_title = "Orientation:{}_Luminosity:{}_Channel:{}_Trial:{}_Start:{}_Stop:{}".format(orientation, luminosity,
                                                                                                channel, trial, start,
                                                                                                stop)
        plt.title(plot_title)
        plt.plot(self.get_dataset_piece(orientation, luminosity, trial, channel)[start:stop], label="LFP signal")
        plt.legend()
        if save:
            plt.savefig(
                "/home/pasca/School/Licenta/Datasets/CER01A50/Plots/{0}.png".format(
                    str(orientation * luminosity) + plot_title))
        if show:
            plt.show()


if __name__ == '__main__':
    dataset = MouseLFP()
    print(dataset.all_lfp_data.shape)
