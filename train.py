import math

from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna

import data

dataset = data.load_dir('data')
maxtime = max(dataset.one_hot['epochseconds'])
mintime = min(dataset.one_hot['epochseconds'])
maxdays = math.ceil((maxtime - mintime)/86400)

def get_columns(dataset, additional_drop=None):
    always_dropped = ['outcome_victory', 'epochseconds', 'uptime']
    if additional_drop is not None:
        always_dropped += additional_drop 
    def incolumn(column):
        for drop in always_dropped:
            if drop in column:
                return True
        return False
    return [column for column in dataset.one_hot.columns if not incolumn(column)]

def select_features(df, features):
    dropped_features = [column for column in df.columns if (column not in features) or features[column] == 0]
    assert 'epochseconds' in dropped_features and 'outcome_victory' not in df.columns
    pruned = df.drop(dropped_features, axis='columns')
    return pruned

def testcv(params):
    print(f"loaded {len(dataset.dataframe)}")
    dfWindow = dataset.one_hot[dataset.one_hot['epochseconds'] > maxtime - (params['limit_days']*86400)]
    print(f"window {len(dfWindow)}")
    assert len(dfWindow)
    dfX = select_features(dfWindow.drop('outcome_victory', axis=1), params)
    dfY = dfWindow['outcome_victory']
    assert len(dfX) == len(dfY)
    d = xgb.DMatrix(dfX, label=dfY)
    print(dfX.columns)
    result = xgb.cv(params, d, num_boost_round=params['num_boost_round'], nfold=20, shuffle=True)
    error = result['test-error-mean'][params['num_boost_round']-1]
    print(error)
    return error

def testtv(params):
    days = 5
    print(f"loaded {len(dataset.dataframe)}")
    dfWindow = dataset.one_hot[dataset.one_hot['epochseconds'] > maxtime - (params['limit_days']*86400)]
    print(f"window {len(dfWindow)}")
    dfTrain = dfWindow[dfWindow['epochseconds'] < maxtime - (days*86400)]
    dfTest = dfWindow[dfWindow['epochseconds'] >= maxtime - (days*86400)]
    assert len(dfTrain) and len(dfTest), f"NO TRAIN OR TEST? limit days: {params['limit_days']}"
    assert len(dfTrain) + len(dfTest) == len(dfWindow)
    dfXTrain = select_features(dfTrain.drop('outcome_victory', axis=1), params)
    dfYTrain = dfTrain['outcome_victory']
    assert len(dfXTrain) == len(dfYTrain)
    dfXTest = select_features(dfTest.drop('outcome_victory', axis=1), params)
    dfYTest = dfTest['outcome_victory']
    assert len(dfXTest) == len(dfYTest)
    dTrain = xgb.DMatrix(dfXTrain, label=dfYTrain)
    dTest = xgb.DMatrix(dfXTest, label=dfYTest)
    evals_result = dict()
    print(dfXTrain.columns)
    result = xgb.train(params, dTrain, num_boost_round=params['num_boost_round'], evals=[(dTest, 'test')], evals_result=evals_result, verbose_eval=False)
    error = evals_result['test']['error'][-1]
    print(error)
    return error

def train_final(params):
    dfTrain = dataset.one_hot
    if 'limit_days' in params.keys() and params['limit_days'] is not None:
        maxtime = max(dfTrain['epochseconds'])
        dfTrain = dfTrain[dfTrain['epochseconds'] > maxtime - (params['limit_days']*86400)]
    dfXTrain = select_features(dfTrain.drop('outcome_victory', axis=1), params)
    dfYTrain = dfTrain['outcome_victory']
    assert len(dfXTrain) == len(dfYTrain)
    print(f"final train with {len(dfTrain)}")
    print("final params", params)
    dTrain = xgb.DMatrix(dfXTrain, label=dfYTrain)
    model = xgb.train(params, dTrain, num_boost_round=params['num_boost_round'])
    return model

def inference_final(inp, history, modelandparams):
    inf_dataframe = data.inference_dataframe(inp, history, dataset.dataframe)
    inf_inp = inf_dataframe.iloc[-1:].drop('outcome_victory', axis=1)
    print(inf_inp.T)
    for basename, model, params in modelandparams:
        dTest = xgb.DMatrix(select_features(inf_inp, params), label=None)
        res = model.predict(dTest)[0]
        if res >= 0.5:
            outcome = 'victory'
        else:
            outcome = 'defeat'
        print(f"{basename}: {outcome} ({res:.3f},{1-res:.3f})")

def getobjective(cv=True, additional_drop=None, select_features=False, no_session_features=False):
    print("getting objective with maxdays:", maxdays)
    def objective(trial):
        columns = get_columns(dataset, additional_drop)
        params = {'eta': trial.suggest_float('eta', 0.01, 100, log=True),
                  'max_depth': trial.suggest_int('max_depth', 1, 20),
                  'alpha': trial.suggest_float('alpha', 0.001, 100, log=True),
                  'lambda': trial.suggest_float('lambda', 0.001, 100, log=True),
                  'gamma': trial.suggest_float('gamma', 0.001, 100, log=True),
                  'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 100, log=True),
                  'max_delta_step': trial.suggest_float('max_delta_step', 0, 10, step=0.25),
                  'subsample': trial.suggest_float('subsample', 0, 1, step=0.05),
                  'objective': 'binary:logistic',
                  'eval_metric': 'error',
                  'verbosity': 0,
                  'num_boost_round': trial.suggest_int('num_boost_round', 10, 1000),
                  'limit_days': trial.suggest_int('limit_days', 10, maxdays) if not cv else trial.suggest_int('limit_days', 10, maxdays),
                 }
        for column in columns:
            if select_features:
                params[column] = trial.suggest_int(column, 0, 1)
            else:
                if no_session_features and 'feature' in column:
                    params[column] = trial.suggest_int(column, 0, 0)
                else:
                    params[column] = trial.suggest_int(column, 1, 1)
        if cv:
            return testcv(params)
        return testtv(params)
    return objective

if __name__ == '__main__':
    params = {'booster': 'gbtree', 'eval_metric': 'error', 'eta':0.3, 'max_depth':2, 'lambda':10}
    #print(get_columns())
    study = optuna.create_study(direction='minimize')
    study.optimize(getobjective(True), n_trials=10000) 
    ##study.optimize(getobjective(False), n_trials=1000)
