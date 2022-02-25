artolearn
---------

artolearn is a collection of scripts for (1) parsing [Artosis](https://twitch.tv/artosis) VODs to extract csvs of game data (e.g., opponent rank, mmr, map, and game outcome), and (2) building and tuning machine learning models for online prediction of game outcomes (gamba gamba MODS OPEN CASINO).

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
Auto-updated 2022-02-24 21:12:15 (560 games)

per-race+rank winrates
----------------------
rank | vs. p | vs. r | vs. t | vs. z | overall
---- | ---- | ---- | ---- | ---- | ---- 
unranked | 54.2% | 33.3% | 58.8% | 40.0% | 50.0% 
a | 46.7% | 62.5% | 47.7% | 52.9% | 49.5% 
b | 85.4% | 100.0% | 78.1% | 87.9% | 85.7% 
c | 100.0% | 100.0% | 100.0% | 80.0% | 92.3% 
d | 100.0% | nan% | nan% | nan% | 100.0% 
s | 23.3% | nan% | 20.0% | 14.3% | 20.3% 
overall | 53.9% | 80.8% | 53.8% | 55.7% | 55.7%

 baseline accuracy from rank alone: `62.5%`

 baseline accuracy from race alone: `55.7%`

 race+rank baseline accuracy: `64.8%`

baseline accuracy from always picking higher mmr player:`66.4%`

map/matchup winrates
--------------------

map | vs. p | vs. r | vs. t | vs. z
------|------|------|------|------
eclipse | 45.8% | 83.3% | 55.6% | 62.2%
largo | 69.0% | 100.0% | 47.8% | 62.2%
polypoid | 55.4% | 66.7% | 62.2% | 56.6%
goodnight | 51.6% | 100.0% | 42.3% | 35.5%
