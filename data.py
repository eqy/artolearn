import datetime
import csv
import os

import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

import constants

def sub(l1, l2):
    return [first - second for first, second in zip(l1, l2)]

def div(l1, l2):
    return [first / (second + 1e-6) for first, second in zip(l1, l2)]

class ResultFeatureTracker(object):
    def __init__(self, pendingasvictory=True):
        self.wins = [0, 0, 0, 0, 0]
        self.losses = [0, 0, 0, 0, 0]
        self.winstreaks = [0, 0, 0, 0, 0]
        self.lossstreaks = [0, 0, 0, 0, 0]
        self.pendingasvictory = pendingasvictory

    def current_features(self):
        return self.winstreaks + self.lossstreaks + div(self.wins, self.losses)

    def update(self, result, race):
        race_to_idx = {'p': 1, 't': 2, 'z': 3, 'r': 4}
        if result == 'victory' or (self.pendingasvictory and result == 'pending'):
            self.wins[0] += 1
            self.wins[race_to_idx[race]] += 1
            self.winstreaks[race_to_idx[race]] += 1
            self.lossstreaks[race_to_idx[race]] = 0
            self.lossstreaks[0] = 0
            result = 'victory'
        else:
            self.losses[0] += 1
            self.losses[race_to_idx[race]] += 1
            self.lossstreaks[race_to_idx[race]] += 1
            self.winstreaks[race_to_idx[race]] = 0
            self.winstreaks[0] = 0
        return result 

def compute_name_features(name):
    name_vowel_count = 0
    name_09_count = 0
    for c in name:
        if c in 'aeiou':
            name_vowel_count += 1
        elif c in '0123456789':
            name_09_count += 1
    name_len = len(name)
    name_features = [name_len,
                     name_vowel_count/max(1, name_len),
                     name_09_count/max(1, name_len)]
    return name_features

def inference_dataframe(inp, history, dataframe):
    result_feature_tracker = ResultFeatureTracker()
    for idx in range(0, len(history), 2):
        result = 'victory' if history[idx+1] == 'w' else 'defeat'
        race = history[idx]
        result_feature_tracker.update(result, race)
    # name_features = compute_name_features(inp[constants.OPPONENT_NAME_IDX])
    curr_row = inp + result_feature_tracker.current_features() # + name_features
    print(curr_row)
    assert len(curr_row) == len(dataframe.columns), f"mismatched schema {len(curr_row)} {len(dataframe.columns)}"
    # This looks expensive but this is how it's done in the docs yikes
    curr_row_df = pd.DataFrame([curr_row], columns=dataframe.columns) 
    out = pd.concat([dataframe, curr_row_df], axis=0, ignore_index=True)
    out = pd.get_dummies(out.drop(labels=['player_a_id', 'player_b_id'], axis=1), drop_first=True, dtype=bool)
    return out

class StreamSessionData(object):
    def __init__(self, filepath, pendingasvictory=True):
        self.filepath = filepath
        assert os.path.exists(filepath) 
        self.result_feature_tracker = ResultFeatureTracker(pendingasvictory)
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
        self.data = list()
        for rawdata in self.rawdata:
            rawdata = list(rawdata)
            oppo_name = rawdata[constants.OPPONENT_NAME_IDX]
            # name_features = compute_name_features(oppo_name)
            current_features = self.result_feature_tracker.current_features().copy()
            oppo_race = rawdata[constants.OPPONENT_RACE_IDX]
            result = rawdata[constants.RESULT_IDX]
            result = self.result_feature_tracker.update(result, oppo_race)
            rawdata[constants.RESULT_IDX] = result
            self.data.append(rawdata + current_features)# + name_features)
            

class Dataset(object):
    def __init__(self, stream_session_data_list):
        self.stream_session_data_list = stream_session_data_list
        self.to_one_hot()

    def to_one_hot(self):
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
    print("baselines...")
    victory_dataframe = dataset.dataframe[dataset.dataframe['outcome'] == 'victory']
    print("global winrate:", len(victory_dataframe)/len(dataset.dataframe))
    opponent_races = dataset.dataframe['player_b_race'].unique()
    opponent_ranks = dataset.dataframe['player_b_rank'].unique()
    lastepochtime = max(dataset.dataframe['epochseconds'])
    text = ("\nAdditional Fun Summary Statistics From Collected Data\n"
     "-----------------------------------------------------\n"
    f"Auto-updated {datetime.datetime.fromtimestamp(lastepochtime)} ({len(dataset.dataframe)} games)\n"
     "\nper-rank winrates\n"
     "-----------------\n")
    total_count = 0
    total = 0
    text += ("Rank | Winrate\n"
             "---- | -------\n")
    for rank in opponent_ranks:
        rank_dataframe = dataset.dataframe[dataset.dataframe['player_b_rank'] == rank]
        victory_dataframe = rank_dataframe[rank_dataframe['outcome'] == 'victory']
        if rank == '':
            rank = 'unranked'
        winrate = len(victory_dataframe)/len(rank_dataframe)
        text += f"{rank} | {winrate*100:.1f}%\n"
        total_count += len(rank_dataframe)
        total += len(rank_dataframe) * max(winrate, 1-winrate)
    text += f"\n baseline accuracy from rank only: `{(total/total_count)*100:.1f}%`\n"
    total_count = 0
    total = 0
    text += ("\nper-race winrates\n"
             "-----------------\n"
             "Rank | Winrate\n"
             "---- | -------\n")
    for race in opponent_races:
        race_dataframe = dataset.dataframe[dataset.dataframe['player_b_race'] == race]
        victory_dataframe = race_dataframe[race_dataframe['outcome'] == 'victory']
        winrate = len(victory_dataframe)/len(race_dataframe)
        text += f"{race} | {winrate*100:.1f}%\n"
        total_count += len(race_dataframe)
        total += len(race_dataframe) * max(winrate, 1-winrate)
    text += f"\n baseline accuracy from race alone: `{(total/total_count)*100:.1f}%`\n"
    total_count = 0
    total = 0
    text += ("\nper-race+rank winrates\n"
             "----------------------\n"
             "Race | Rank | Winrate \n"
             "---- | ---- | ------- \n")
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
            text += f"{printrank} | {race} | {winrate*100:.1f}%\n"
    text += f"\n race+rank baseline accuracy: `{(total/total_count)*100:.1f}%`\n"
    pd.set_option("display.max_rows", 300, "display.max_columns", 4)
    print(dataset.dataframe)
    placeholder = [1600000.0,
                   'artosis', 'a', 2043.0, 't',
                   'asdffjiej1290381209470', 's', 2390.0, 'z',
                   'polypoid', 24.0, 'low', 3000.0, 'defeat']
    print(len(dataset.dataframe.columns))
    print(len(dataset.dataframe))
    out_df = inference_dataframe(placeholder, 'zlzlzl', dataset.dataframe)
    assert len(out_df) == len(dataset.dataframe) + 1
    #print(out_df)
    assert len(dataset.one_hot.columns) == len(out_df.columns)

    available_df = dataset.dataframe[(dataset.dataframe['player_a_mmr'] > 0) & (dataset.dataframe['player_b_mmr'] > 0)]
    total_available = len(available_df)
    correct_win = available_df[(available_df['player_a_mmr'] > available_df['player_b_mmr']) & ((available_df['outcome'] == 'victory') | (available_df['outcome'] == 'pending'))]
    correct_lose  = available_df[(available_df['player_a_mmr'] < available_df['player_b_mmr']) & (available_df['outcome'] == 'defeat')]
    text += f"\nbaseline accuracy from always picking higher mmr player:`{100*(len(correct_win)+len(correct_lose))/total_available:.1f}%`\n"

    lines = list()
    with open('README_base.md', 'r') as f:
        for line in f.readlines():
            lines.append(line)
    lines += text
    with open('README.md', 'w') as f:
        f.writelines(lines) 
    print("OK")
        
            
if __name__ == '__main__':
    test()
