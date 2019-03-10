import csv
import json
import numpy as np
import os


class LFPDataset:

    def __init__(self, dataset_path, saved_as_npy=True):
        self.description_file_path = dataset_path
        if saved_as_npy:
            self.load_from_npy(dataset_path)
        else:
            lfp_file_data = LFPDataset.load_from_npy(dataset_path)
            self.bin_file_names = lfp_file_data.get('bin_file_names')
            self.trial_length = lfp_file_data.get('trial_length')
            self.total_length = lfp_file_data.get('total_length')
            self.sampling_frequency = lfp_file_data.get('sampling_frequency')
            self.number_of_lfp_files = lfp_file_data.get('number_of_lfp_files')
            self.ldf_file_version = lfp_file_data.get('ldf_file_version')
            self.stimulus_condition_file = lfp_file_data.get('stimulus_condition_file')
            self.number_of_conditions = lfp_file_data.get('number_of_conditions')
            self.trials_per_condition = lfp_file_data.get('trials_per_condition')
            self.event_codes_file_path = lfp_file_data.get('event_codes_file')
            self.event_timestamps_file_path = lfp_file_data.get('event_timestamps_file')
            self.event_codes = self._parse_event_codes(lfp_file_data.get('event_codes_file'))
            self.event_timestamps = self._parse_event_timestamps(lfp_file_data.get('event_timestamps_file'))
            self.stimulus_conditions = self._parse_stimulus_data(lfp_file_data.get('stimulus_condition_file'))
            self.channels = self._parse_channels_data(lfp_file_data.get('bin_file_names'))
            self.stimulus_on_at, self.stimulus_off_at = self._parse_stimulus_on_off(lfp_file_data)

        self.nr_channels = len(self.channels)

    def _parse_stimulus_data(self, condition_file_path):
        with open(os.path.join(os.path.dirname(self.description_file_path), condition_file_path),
                  'r') as f:
            if self.ldf_file_version == '1.0':
                return [int(st) for st in f.read().split('\n')]
            elif self.ldf_file_version == '1.1':
                csv_reader = csv.reader(f, delimiter=',')
                self.condition_refresh_rate = float(next(csv_reader)[1])
                self.experiment_duration = float(next(csv_reader)[1])
                col_descs = next(csv_reader)
                conditions = []
                for row in csv_reader:
                    cond = {}
                    for index, col in enumerate(col_descs):
                        cond[col] = row[index]
                    conditions.append(cond)
                return conditions

    def _parse_channels_data(self, channels_paths):
        return np.array(
            [np.fromfile(open(os.path.join(os.path.dirname(self.description_file_path), channel_path), 'rb'),
                         np.float32) for channel_path in channels_paths])

    def _parse_event_codes(self, file_path):
        if file_path is None:
            return None
        return np.fromfile(open(os.path.join(os.path.dirname(self.description_file_path), file_path), 'rb'),
                           np.int32)

    def _parse_event_timestamps(self, file_path):
        if file_path is None:
            return None
        return np.fromfile(open(os.path.join(os.path.dirname(self.description_file_path), file_path), 'rb'),
                           np.int32)

    def _parse_stimulus_on_off(self, lfp_description):
        if self.ldf_file_version == '1.0':
            return lfp_description.get('stimulus_on_at'), lfp_description.get('stimulus_off_at')
        elif self.ldf_file_version == '1.1':
            stimulus_on_at = []
            stimulus_off_at = []
            for index, event in enumerate(self.event_codes):
                if event == 129:
                    stimulus_on_at.append(self.event_timestamps[index])
                if event == 150:
                    stimulus_off_at.append(self.event_timestamps[index])
            return stimulus_on_at, stimulus_off_at

    def save_as_npy(self, path):
        np.save(path, vars(self))

    def load_from_npy(self, path):
        lfp_file_data = np.load(path).item()
        for prop, val in lfp_file_data.items():
            setattr(self, prop, val)

    @staticmethod
    def _parse_description(description_file_path):
        with open(description_file_path, 'r') as f:
            lfp_description = json.loads(f.read())
        return lfp_description
