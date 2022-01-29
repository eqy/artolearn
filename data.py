import csv
import os

import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

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
                rawdata[constants.RESULT_IDX] = 'victory'
                # outcome = 1
            else:
                losses[0] += 1
                losses[race_to_idx[oppo_race]] += 1
                winstreaks[race_to_idx[oppo_race]] = 0
                winstreaks[0] = 0
                # outcome = 0
            rawdata = list(rawdata)
            #rawdata[constants.ARTOSIS_NAME_IDX] = None
            #rawdata[constants.OPPONENT_NAME_IDX] = None
            self.data.append(rawdata + wins + losses + sub(wins, losses) + winstreaks + lossstreaks)

class Dataset(object):
    def __init__(self, stream_session_data_list):
        self.stream_session_data_list = stream_session_data_list
        self.to_one_hot()

    def to_one_hot(self):
        #numerical_data = list()
        #categorical_data = list()
        #result = list()
        #for stream_session_data in self.stream_session_data_list:
        #    for row in stream_session_data.data:
        #        numerical, categorical = splitnumerical(row[:-1])
        #        numerical_data.append(numerical)
        #        categorical_data.append(categorical)
        #        result.append(row[-1])
        #self.encoder = OneHotEncoder(drop='first')
        #self.encoder.fit(categorical_data)
        #self.categorical_encoded = self.encoder.transform(categorical_data).toarray()
        #self.data = np.concatenate((np.array(numerical_data), self.categorical_encoded), axis=1)
        #self.labels = np.array(result)
        columns = len(self.stream_session_data_list[0].data[0])
        data_lists = [list() for _ in range(columns)]
        for stream_session_data in self.stream_session_data_list:
            for row in stream_session_data.data:
                assert len(row) == columns
                for idx, value in enumerate(row):
                    try:
                        value = float(value)
                    except ValueError:
                        pass     
                    data_lists[idx].append(value)
        column_labels = constants.SCHEMA + [f'feature{i}' for i in range(0, columns - constants.SCHEMA_LEN)]
        data_dict = {label:data_lists[i] for i,label in enumerate(column_labels)}
        self.dataframe = pd.DataFrame(data_dict)
        self.labels = pd.get_dummies(self.dataframe['outcome'], drop_first=True, dtype=bool)
        self.one_hot = pd.get_dummies(self.dataframe.drop(labels=['player_a_id', 'player_b_id'], axis=1), drop_first=True, dtype=bool)

def load_dir(path):
    rawdataset = list()
    for dirpath, _, filenames in os.walk('data'):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == '.csv':
                data = StreamSessionData(os.path.join(dirpath, filename))
                rawdataset.append(data)
    return Dataset(rawdataset)

def test():
    data = StreamSessionData('data/2021-12-23.csv')
    dataset = load_dir('data')
    #print(dataset.data)
    #print(dataset.labels)
    print("baselines...")
    victory_dataframe = dataset.dataframe[dataset.dataframe['outcome'] == 'victory']
    print("global winrate:", len(victory_dataframe)/len(dataset.dataframe))
    opponent_races = dataset.dataframe['player_b_race'].unique()
    opponent_ranks = dataset.dataframe['player_b_rank'].unique()
    print("per rank winrates")
    total_count = 0
    total = 0
    for rank in opponent_ranks:
        rank_dataframe = dataset.dataframe[dataset.dataframe['player_b_rank'] == rank]
        victory_dataframe = rank_dataframe[rank_dataframe['outcome'] == 'victory']
        if rank == '':
            rank = 'unranked'
        winrate = len(victory_dataframe)/len(rank_dataframe)
        print(f"{rank}: {winrate}")
        total_count += len(rank_dataframe)
        total += len(rank_dataframe) * max(winrate, 1-winrate)
    print(f"rank baseline: {total/total_count} {total_count}")
    total_count = 0
    total = 0
    print("per race winrates")
    for race in opponent_races:
        race_dataframe = dataset.dataframe[dataset.dataframe['player_b_race'] == race]
        victory_dataframe = race_dataframe[race_dataframe['outcome'] == 'victory']
        winrate = len(victory_dataframe)/len(race_dataframe)
        print(f"{race}: {winrate}")
        total_count += len(race_dataframe)
        total += len(race_dataframe) * max(winrate, 1-winrate)
    print(f"race baseline: {total/total_count} {total_count}")
    total_count = 0
    total = 0
    print("per race+rank winrates")
    for rank in opponent_ranks:
        for race in opponent_races:
            rank_dataframe = dataset.dataframe[dataset.dataframe['player_b_rank'] == rank]
            race_dataframe = rank_dataframe[rank_dataframe['player_b_race'] == race]
            if len(race_dataframe) == 0:
                continue
            victory_dataframe = race_dataframe[race_dataframe['outcome'] == 'victory']
            printrank = rank
            if rank == '':
                printrank = 'unranked'
            winrate = len(victory_dataframe)/len(race_dataframe)
            total_count += len(race_dataframe)
            total += len(race_dataframe) * max(winrate, 1-winrate) 
            print(f"{printrank} {race}: {winrate}")
    print(f"race+rank baseline: {total/total_count} {total_count}")
    pd.set_option("display.max_rows", 300, "display.max_columns", 4)
    print(dataset.dataframe)
    print("OK")
            
if __name__ == '__main__':
    test()
