import csv
import os

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import constants

def sub(l1, l2):
    return [first - second for first, second in zip(l1, l2)]

def splitnumerical(row):
    numerical = list()
    categorical = list()
    for idx, item in enumerate(row):
        if item is not None:
            try:
                value = float(item)
                numerical.append(value)
            except ValueError as _:
                categorical.append(item)
    return numerical, categorical

class StreamSessionData(object):
    def __init__(self, filepath, pendingasvictory=True):
        self.filepath = filepath
        assert os.path.exists(filepath) 
        self.pendingasvictory = pendingasvictory
        self.rawdata = None
        self.load()
        self.compute_features()

    def load(self):
        self.rawdata = list()
        with open(self.filepath, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                curr_rawdata = row[:constants.SCHEMA_LEN]
                curr_rawdata = [item.lower() if isinstance(item, str) else item for item in curr_rawdata]
                if curr_rawdata[constants.ARTOSIS_MMR_IDX] == '':
                    curr_rawdata[constants.ARTOSIS_MMR_IDX] = '0'
                if curr_rawdata[constants.OPPONENT_MMR_IDX] == '':
                    curr_rawdata[constants.OPPONENT_MMR_IDX] = '0'
                if curr_rawdata[constants.TURNRATE_IDX] == '':
                    curr_rawdata[constants.TURNRATE_IDX] = '0'
                self.rawdata.append(curr_rawdata)

    def compute_features(self):
        assert self.rawdata is not None
        #total, p, t, z, r
        wins = [0, 0, 0, 0, 0]
        losses = [0, 0, 0, 0, 0]
        winstreaks = [0, 0, 0, 0, 0]
        lossstreaks = [0, 0, 0, 0, 0]
        race_to_idx = {'p': 1, 't': 2, 'z': 3, 'r': 4}
        self.data = list()
        for rawdata in self.rawdata:
            oppo_race = rawdata[constants.OPPONENT_RACE_IDX]
            result = rawdata[constants.RESULT_IDX]
            if result == 'victory' or (self.pendingasvictory and result == 'pending'):
                wins[0] += 1
                wins[race_to_idx[oppo_race]] += 1
                winstreaks[race_to_idx[oppo_race]] += 1
                lossstreaks[race_to_idx[oppo_race]] = 0
                lossstreaks[0] = 0
                outcome = 1
            else:
                losses[0] += 1
                losses[race_to_idx[oppo_race]] += 1
                winstreaks[race_to_idx[oppo_race]] = 0
                winstreaks[0] = 0
                outcome = 0
            rawdata = list(rawdata)
            rawdata[constants.ARTOSIS_NAME_IDX] = None
            rawdata[constants.OPPONENT_NAME_IDX] = None
            self.data.append(rawdata[:constants.RESULT_IDX] + wins + losses + sub(wins, losses) + winstreaks + lossstreaks + [outcome])

class Dataset(object):
    def __init__(self, stream_session_data_list):
        self.stream_session_data_list = stream_session_data_list
        self.to_one_hot()

    def to_one_hot(self):
        numerical_data = list()
        categorical_data = list()
        result = list()
        for stream_session_data in self.stream_session_data_list:
            for row in stream_session_data.data:
                numerical, categorical = splitnumerical(row[:-1])
                numerical_data.append(numerical)
                categorical_data.append(categorical)
                result.append(row[-1])
        self.encoder = OneHotEncoder(drop='first')
        self.encoder.fit(categorical_data)
        self.categorical_encoded = self.encoder.transform(categorical_data).toarray()
        self.data = np.concatenate((np.array(numerical_data), self.categorical_encoded), axis=1)
        self.labels = np.array(result)

def test():
    data = StreamSessionData('data/2021-12-23.csv')
    rawdataset = list()
    for dirpath, _, filenames in os.walk('data'):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == '.csv':
                data = StreamSessionData(os.path.join(dirpath, filename))
                rawdataset.append(data)
    dataset = Dataset(rawdataset)
    print(dataset.data)
    print(dataset.labels)
    print("OK")
            
if __name__ == '__main__':
    test()
