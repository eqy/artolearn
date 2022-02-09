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
Auto-updated 2022-02-08 19:23:18 (404 games)

per-rank winrates
-----------------
Rank | Winrate
---- | -------
unranked | 62.7%
a | 49.8%
b | 84.3%
c | 0.0%
d | 100.0%
s | 16.3%

 baseline accuracy from rank only: `63.4%`

per-race winrates
-----------------
Rank | Winrate
---- | -------
p | 54.9%
r | 85.7%
t | 53.8%
z | 52.9%

 baseline accuracy from race alone: `55.7%`

per-race+rank winrates
----------------------
Rank | vs. p | vs. r | vs. t | vs. z
---- | ---- | ---- | ---- | ---- 
unranked | 70.8% | 50.0% | 66.7% | 52.6% 
a | 48.9% | 83.3% | 48.9% | 48.5% 
b | 84.8% | 100.0% | 70.8% | 90.5% 
c | nan% | nan% | nan% | 0.0% 
d | 100.0% | nan% | nan% | nan% 
s | 20.0% | nan% | 12.5% | 10.0% 

 race+rank baseline accuracy: `64.4%`

baseline accuracy from always picking higher mmr player:`66.1%`

map/matchup winrates
--------------------

map | vs. p | vs. r | vs. t | vs. z
------|------|------|------|------
eclipse | 41.5% | 80.0% | 60.0% | 60.0%
largo | 70.6% | 100.0% | 42.9% | 57.6%
polypoid | 59.6% | 80.0% | 60.6% | 55.3%
goodnight | 59.1% | 100.0% | 42.1% | 34.8%
