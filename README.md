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
Auto-updated 2022-02-03 21:34:08 (370 games)

per-rank winrates
-----------------
Rank | Winrate
---- | -------
a | 50.0%
b | 84.5%
unranked | 62.5%
s | 18.4%
c | 0.0%
d | 100.0%

 baseline accuracy from rank only: `63.2%`

per-race winrates
-----------------
Rank | Winrate
---- | -------
p | 55.7%
z | 53.3%
t | 54.5%
r | 89.5%

 baseline accuracy from race alone: `56.5%`

per-race+rank winrates
----------------------
Race | Rank | Winrate 
---- | ---- | ------- 
a | p | 48.1%
a | z | 50.0%
a | t | 50.0%
a | r | 80.0%
b | p | 87.1%
b | z | 89.5%
b | t | 69.6%
b | r | 100.0%
unranked | p | 69.6%
unranked | z | 50.0%
unranked | t | 66.7%
unranked | r | 66.7%
s | p | 22.7%
s | z | 11.1%
s | t | 14.3%
c | z | 0.0%
d | p | 100.0%

 race+rank baseline accuracy: `64.1%`

baseline accuracy from always picking higher mmr player:`65.2%`
