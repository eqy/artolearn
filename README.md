artolearn
---------

artolean is a collection of scripts for (1) parsing [Artosis](https://twitch.tv/artosis) VODs to extract csvs of game data (e.g., opponent rank, mmr, map, and game outcome), and (2) building and tuning machine learning models for online prediction of game outcomes (gamba gamba MODS OPEN CASINO).

Data Extraction
---------------
Game data is extracted via OpenCV, where still frames of games are matched against reference game scenes to determine the current stream VOD state (e.g., match-found (where MMRs are displayed), in-game, post-game (where outcomes are displayed), and so on.
Once frames are matched, additional OCR is performed if necessary to extract information such as the current map, the MMRs of players, and so on.
The "heavy lifting" is done by simple nearest-neighbor matching to reference images for scenes and a few other scenarios (e.g., determining the race selected by each player in the match screen).
Additional heavy lifting is done by [pytesseract](https://pypi.org/project/pytesseract/) for OCR, with some intermediate steps in OpenCV for thresholding and other operations to make the text more OCR-friendly.

The data extraction CLI is provided by `video2csv.py`.
Extracted data csvs are in `data/`.
Note that for copyright reasons, the reference scene images are omitted from
this repository.

Modeling and Prediction
-----------------------
As the extracted data contains a mixture of numerical (player MMRs) and categorical features (map, races chosen by players, ...) the current modeling uses [xgboost](https://xgboost.readthedocs.io/en/stable/) with [optuna](https://optuna.org/) for hyperparameter tuning.

Prediction experiments are still at a very early and highly experimental stage, with many changes pending due to some tricky gotchas associated with a time-dependent classification task.
For example, it may be useful to include time-dependent features such as the current stream uptime and win/loss streaks within a stream, but this introduces potential leaks of data labels.
Similarly, it may be useful to limit the time horizon of included datapoints to account for MMR inflation or metagame shifts (e.g., by pruning very old game results from the dataset), but treating the time horizon as a hyperparameter may increase overfitting.

The tuning CLI is provided by `csv2model.py`.
The live inference CLI is provided by `infer.py`.

Additional Fun Summary Statistics From Collected Data
-----------------------------------------------------
Auto-updated 2022-02-07 20:39:24 (395 games)

per-rank winrates
-----------------
Rank | Winrate
---- | -------
unranked | 62.7%
a | 51.2%
b | 85.1%
c | 0.0%
d | 100.0%
s | 16.7%

 baseline accuracy from rank only: `64.1%`

per-race winrates
-----------------
Rank | Winrate
---- | -------
p | 56.5%
r | 85.7%
t | 54.4%
z | 53.4%

 baseline accuracy from race alone: `56.7%`

per-race+rank winrates
----------------------
Race | Rank | Winrate 
---- | ---- | ------- 
unranked | p | 70.8%
unranked | r | 50.0%
unranked | t | 66.7%
unranked | z | 52.6%
a | p | 50.6%
a | r | 83.3%
a | t | 50.0%
a | z | 50.0%
b | p | 87.5%
b | r | 100.0%
b | t | 70.8%
b | z | 90.0%
c | z | 0.0%
d | p | 100.0%
s | p | 20.8%
s | t | 12.5%
s | z | 10.0%

 race+rank baseline accuracy: `64.1%`

baseline accuracy from always picking higher mmr player:`66.1%`

map | vs. p | vs. r | vs. t | vs. z
------|------|------|------|------
eclipse | 43.5% | vs. 80.0% | vs. 60.0% | vs. 60.0%
largo | 72.7% | vs. 100.0% | vs. 46.2% | vs. 57.6%
polypoid | 59.6% | vs. 80.0% | vs. 60.6% | vs. 54.1%
goodnight | 61.9% | vs. 100.0% | vs. 42.1% | vs. 38.1%
