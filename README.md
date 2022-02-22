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
Auto-updated 2022-02-21 21:30:12 (511 games)

per-race+rank winrates
----------------------
rank | vs. p | vs. r | vs. t | vs. z | overall
---- | ---- | ---- | ---- | ---- | ---- 
unranked | 54.5% | 0.0% | 64.3% | 33.3% | 48.2% 
a | 49.5% | 62.5% | 48.4% | 52.2% | 50.6% 
b | 84.4% | 100.0% | 77.4% | 87.1% | 84.9% 
c | 100.0% | 100.0% | 100.0% | 80.0% | 92.3% 
d | 100.0% | nan% | nan% | nan% | 100.0% 
s | 22.2% | nan% | 16.7% | 16.7% | 19.6% 
overall | 55.5% | 79.2% | 55.0% | 55.1% | 56.4%

 baseline accuracy from rank alone: `62.8%`

 baseline accuracy from race alone: `56.4%`

 race+rank baseline accuracy: `64.6%`

baseline accuracy from always picking higher mmr player:`65.9%`

map/matchup winrates
--------------------

map | vs. p | vs. r | vs. t | vs. z
------|------|------|------|------
eclipse | 44.7% | 80.0% | 57.1% | 60.0%
largo | 67.5% | 100.0% | 50.0% | 61.9%
polypoid | 60.6% | 66.7% | 62.8% | 58.3%
goodnight | 55.6% | 100.0% | 40.9% | 32.1%
