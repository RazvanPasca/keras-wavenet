import json
import numpy as np
import os


class LFPDataset:
    def __init__(self, data_path):
        with open(data_path, 'r+') as f:
            lfp_data = json.loads(f.read())
        with open(os.path.join(os.path.dirname(data_path), lfp_data['stimulus_condition_file']), 'r+') as f:
            lfp_data['stimulusOrder'] = [int(st) for st in f.read().split('\n')]
        lfp_data['channels'] = np.array(
            [np.fromfile(open(os.path.join(os.path.dirname(data_path), file), 'rb'), np.float32) for file in
             lfp_data['bin_file_names']])
        self.file_path = data_path
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
        self.lfp_data = {1: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                         2: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length)),
                         3: np.zeros((self.trials_per_condition, self.number_of_lfp_files, self.trial_length))}
        cur_element = [0, 0, 0, 0]
        for i in range(0, self.channels.shape[1] // self.trial_length):
            condition_number = self.stimulusOrder[i]
            self.lfp_data[condition_number][cur_element[condition_number], :, :] = self.channels[:,
                                                                                   (i * self.trial_length):((
                                                                                                                    i * self.trial_length) + self.trial_length)]
            cur_element[condition_number] += 1
        self.trial_length = 1000  # lfp_data['trial_length']
        min_val = min(np.min(self.lfp_data[1]), np.min(self.lfp_data[2]), np.min(self.lfp_data[3]))
        max_val = max(np.max(self.lfp_data[1]), np.max(self.lfp_data[2]), np.max(self.lfp_data[3]))
        self.values_range = min_val, max_val
        self.train = [0][:self.trial_length]  # 0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 16, 17, 19]
        self.validation = [5][:self.trial_length]  # , 10, 15]
        self.test = [7][:self.trial_length]  # , 13, 18]
        self.train_length = len(self.train) * self.trial_length
        self.validation_length = len(self.validation) * self.trial_length
        self.test_length = len(self.test) * self.trial_length

    def train_frame_generator(self, frame_size, batch_size, output_transform):
        x = []
        y = []
        while 1:
            target_series = self.lfp_data[1][np.random.choice(self.train)][0]
            batch_start = np.random.choice(range(0, self.trial_length - frame_size - 1))
            frame = target_series[batch_start:batch_start + frame_size]
            temp = target_series[batch_start + frame_size]
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
            target_series = self.lfp_data[1][np.random.choice(self.validation)][0]
            batch_start = np.random.choice(range(0, self.trial_length - frame_size - 1))
            frame = target_series[batch_start:batch_start + frame_size]
            temp = target_series[batch_start + frame_size]
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
            target_series = self.lfp_data[1][np.random.choice(self.test)][0]
            batch_start = np.random.choice(range(0, self.trial_length - frame_size - 1))
            frame = target_series[batch_start:batch_start + frame_size]
            temp = target_series[batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(output_transform(temp))
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def get_validation_set(self, movie, trail, channel):
        return self.lfp_data[movie][trail][channel]
