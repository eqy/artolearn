from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna

import data

dataset = data.load_dir('data')

def get_columns(dataset):
    return [column for column in dataset.one_hot.columns if not ('outcome' in column)]

def select_features(df, params):
    dropped_features = [column for column in params['features'].keys() if params['features'][column] == 0]
    pruned = df.drop(dropped_features, axis='columns')
    return pruned

def testcv(params, num_boost_round):
    print(f"loaded {len(dataset.dataframe)}")
    dfX = select_features(dataset.one_hot.drop('outcome_victory', axis=1), params)
    dfY = dataset.one_hot['outcome_victory']
    d = xgb.DMatrix(dfX, label=dfY)
    result = xgb.cv(params, d, num_boost_round=num_boost_round, nfold=20, shuffle=True)
    error = result['test-error-mean'][num_boost_round-1]
    print(result)
    print(error)
    return error

def testtv(params, num_boost_round, limit_days):
    days = 3
    print(f"loaded {len(dataset.dataframe)}")
    maxtime = max(dataset.one_hot['time'])
    dfWindow = dataset.one_hot[dataset.one_hot['time'] > maxtime - limit_days*86400]
    dfTrain = dfWindow[dfWindow['time'] < maxtime - days*86400]
    dfTest = dfWindow[dfWindow['time'] >= maxtime - days*86400]

    dfTrain = select_features(dfTrain, params)
    dfTest = select_features(dfTest, params)

    assert len(dfTrain) + len(dfTest) == len(dfWindow)
    dfXTrain = dfTrain.drop('outcome_victory', axis=1)
    dfYTrain = dfTrain['outcome_victory']
    dfXTest = dfTest.drop('outcome_victory', axis=1)
    dfYTest = dfTest['outcome_victory']
    dTrain = xgb.DMatrix(dfXTrain, label=dfYTrain)
    dTest = xgb.DMatrix(dfXTest, label=dfYTest)
    evals_result = dict()
    result = xgb.train(params, dTrain, num_boost_round=num_boost_round, evals=[(dTest, 'test')], evals_result=evals_result, verbose_eval=False)
    error = evals_result['test']['error'][-1]
    print(error)
    return error

def getobjective(cv=True):
    def objective(trial):
        columns = get_columns(dataset)
        params = {'eta': trial.suggest_float('eta', 0.01, 100, log=True),
                  'max_depth': trial.suggest_int('max_depth', 1, 20),
                  'alpha': trial.suggest_float('alpha', 0.001, 100, log=True),
                  'lambda': trial.suggest_float('lambda', 0.001, 100, log=True),
                  'gamma': trial.suggest_float('gamma', 0.001, 100, log=True),
                  'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 100, log=True),
                  'max_delta_step': trial.suggest_float('max_delta_step', 0, 10, step=0.25),
                  'subsample': trial.suggest_float('subsample', 0, 1, step=0.05),
                  'eval_metric': 'error',
                  'verbosity': 0,
                  'features': {column: trial.suggest_int(column, 0, 1) for column in columns}
                 }
        num_boost_rounds = trial.suggest_int('num_boost_round', 10, 1000)
        if not cv:
            return testcv(params, num_boost_rounds)
        return testtv(params, num_boost_rounds, trial.suggest_int('limit_days', 5, 50))
    return objective

if __name__ == '__main__':
    params = {'booster': 'gbtree', 'eval_metric': 'error', 'eta':0.3, 'max_depth':2, 'lambda':10}
    #print(get_columns())
    study = optuna.create_study(direction='minimize')
    study.optimize(getobjective(True), n_trials=2000) 
    ##study.optimize(getobjective(False), n_trials=2000)
