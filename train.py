from sklearn.model_selection import train_test_split
import xgboost as xgb

import data

def testcv():
    dataset = data.load_dir('data')
    print(f"loaded {len(dataset.dataframe)}")
    dfX = dataset.one_hot.drop('outcome_victory', axis=1)
    dfY = dataset.one_hot['outcome_victory']
    d = xgb.DMatrix(dfX, label=dfY)
    params = {'booster': 'gbtree', 'eval_metric': 'error'}
    result = xgb.cv(params, d, num_boost_round=20, nfold=30, shuffle=True)
    print(result)

def testtv():
    dataset = data.load_dir('data')
    limit_days = 60
    days = 3
    print(f"loaded {len(dataset.dataframe)}")
    maxtime = max(dataset.one_hot['time'])
    dfWindow = dataset.one_hot[dataset.one_hot['time'] > maxtime - limit_days*86400]
    dfTrain = dfWindow[dfWindow['time'] < maxtime - days*86400]
    dfTest = dfWindow[dfWindow['time'] >= maxtime - days*86400]
    assert len(dfTrain) + len(dfTest) == len(dfWindow)
    print(len(dfTrain))
    print(len(dfTest))
    dfXTrain = dfTrain.drop('outcome_victory', axis=1)
    dfYTrain = dfTrain['outcome_victory']
    dfXTest = dfTest.drop('outcome_victory', axis=1)
    dfYTest = dfTest['outcome_victory']
    dTrain = xgb.DMatrix(dfXTrain, label=dfYTrain)
    dTest = xgb.DMatrix(dfXTest, label=dfYTest)
    params = {'booster': 'gbtree', 'eval_metric': 'error'}
    evals_result = dict()
    result = xgb.train(params, dTrain, num_boost_round=20, evals=[(dTest, 'test')], evals_result=evals_result)
    print(evals_result)

if __name__ == '__main__':
    testcv()
    testtv()
