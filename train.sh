#!/bin/bash
python3 csv2model.py -o test -t 5000 --time-validation --additional-drop player_a_rank,player_a_mmr --no-session-features
python3 csv2model.py -o test -t 5000 --time-validation --additional-drop player_a_rank,player_a_mmr
python3 csv2model.py -o test -t 2000 --cross-validation --additional-drop player_a_rank,player_a_mmr --no-session-features
python3 csv2model.py -o test -t 2000 --cross-validation --additional-drop player_a_rank,player_a_mmr
