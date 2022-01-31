import argparse
import datetime

import joblib
import optuna

import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='output file prefix', required=True)
    parser.add_argument('-t', '--trials', help='number of trials', required=True, type=int)
    parser.add_argument('--cross-validation', help='cross-validation mode', action='store_true')
    parser.add_argument('--time-validation', help='time-validation mode', action='store_true')
    parser.add_argument('--additional-drop', help='comma-separated list of additional features to drop', type=str, default='')
    parser.add_argument('--select-features', help='whether to select features', action='store_true')
    parser.add_argument('--no-session-features', help='leave out session feautres', action='store_true')
    args = parser.parse_args()
    assert args.cross_validation ^ args.time_validation, "cross-validation or time-validation must be picked"
    validation_type = 'cross_validation' if args.cross_validation else 'time_validation'
    name_prefix = f"{args.output}_{validation_type}_{args.trials}_{args.additional_drop}_{datetime.date.today()}"

    study = optuna.create_study(direction='minimize')
    additional_drop = args.additional_drop.split(',') if args.additional_drop != '' else None
    columns = train.get_columns(train.dataset)
    for drop in additional_drop:
        assert drop in columns
    study.optimize(train.getobjective(args.cross_validation, additional_drop, args.select_features, args.no_session_features), n_trials=args.trials)
    params = study.best_trial.params
    print(params)
    params['eval_metric'] = 'error'
    joblib.dump(study, f"{name_prefix}.pkl")
    model = train.train_final(params)
    model.save_model(f"{name_prefix}.model") 


if __name__ == '__main__':
    main()
