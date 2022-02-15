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
Auto-updated 2022-02-13 19:53:24 (430 games)

per-race+rank winrates
----------------------
rank | vs. p | vs. r | vs. t | vs. z | overall
---- | ---- | ---- | ---- | ---- | ---- 
unranked | 60.0% | 0.0% | 66.7% | 37.5% | 52.0% 
a | 48.9% | 71.4% | 48.0% | 50.0% | 49.8% 
b | 86.1% | 100.0% | 73.1% | 90.9% | 85.3% 
c | 100.0% | 100.0% | 100.0% | 80.0% | 92.3% 
d | 100.0% | nan% | nan% | nan% | 100.0% 
s | 20.0% | nan% | 22.2% | 10.0% | 18.2% 
overall | 55.2% | 81.8% | 55.1% | 53.5% | 56.0%

 baseline accuracy from rank alone: `62.8%`

 baseline accuracy from race alone: `56.0%`

 race+rank baseline accuracy: `64.9%`

baseline accuracy from always picking higher mmr player:`66.1%`

map/matchup winrates
--------------------

map | vs. p | vs. r | vs. t | vs. z
------|------|------|------|------
eclipse | 42.4% | 80.0% | 63.0% | 58.6%
largo | 69.4% | 100.0% | 43.8% | 58.8%
polypoid | 60.0% | 72.7% | 61.1% | 57.1%
goodnight | 58.3% | 100.0% | 42.1% | 33.3%
