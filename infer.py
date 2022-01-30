import argparse
import joblib
import os
import time

import xgboost as xgb

import constants
import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artommr', type=float, help='artosis mmr', default=0)
    parser.add_argument('--oppommr', type=float, help='opponent mmr', default=0)
    parser.add_argument('--artorace', type=str, help='artosis race', default='t')
    parser.add_argument('--opporace', type=str, help='opponent race', required=True)
    parser.add_argument('--artorank', type=str, help='artosis rank', default='')
    parser.add_argument('--opporank', type=str, help='opponent rank', default='')
    parser.add_argument('--opponame', type=str, help='opponent name', default='')
    parser.add_argument('--history', type=str, help="match history e.g., zwpltw", default='')
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--turnrate', type=float, default=0)
    parser.add_argument('--latency', type=str, default='')
    parser.add_argument('--uptime', type=float, required=True)
    
    args = parser.parse_args()
    if args.artommr > 1800:
        assert len(args.artorank)
    if args.oppommr > 1800:
        assert len(args.opporank)
    assert args.artorace in 'ptzr', "unknown artosis race"
    assert args.opporace in 'ptzr', "unknown opponent race"
    if args.artorace != 't':
        print("WARNING: artosis race not terran, are you sure?")
    assert args.artommr < 3000
    assert args.oppommr < 3000
   
    map_values = set()
    for _, value in constants.MAP_PATTERNS.items():
        map_values.add(value)
 
    assert args.map in map_values, "unknown map"

    inp = [time.time(),
            'artosis'   , args.artorank, args.artommr, args.artorace,
            args.opponame, args.opporank, args.oppommr, args.opporace,
            args.map, args.turnrate, args.latency, args.uptime, 'defeat']
    assert len(args.history) % 2 == 0
    modelandparams = list()
    for dirpath, _, filenames in os.walk('.'):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == '.model':
                assert os.path.exists(os.path.join(dirpath, basename + '.pkl'))
                booster = xgb.Booster()
                booster.load_model(os.path.join(dirpath, filename))
                params = joblib.load(os.path.join(dirpath, basename + '.pkl')).best_trial.params
                print("loaded: ", filename)
                modelandparams.append((basename, booster, params))
    print(inp)
    outcome = train.inference_final(inp, args.history, modelandparams)

if __name__ == '__main__':
    main()
