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
Auto-updated 2022-01-31 21:30:30 (328 games)

per-rank winrates
-----------------
Rank | Winrate
---- | -------
a | 50.9%
b | 81.4%
unranked | 58.7%
s | 19.4%
d | 100.0%

 baseline accuracy from rank only: `61.9%`

per-race winrates
-----------------
Rank | Winrate
---- | -------
p | 54.9%
z | 52.2%
t | 53.2%
r | 92.3%

 baseline accuracy from race alone: `55.2%`

per-race+rank winrates
----------------------
Race | Rank | Winrate 
---- | ---- | ------- 
a | p | 49.3%
a | z | 50.9%
a | t | 50.0%
a | r | 80.0%
b | p | 85.7%
b | z | 86.7%
b | t | 65.0%
b | r | 100.0%
unranked | p | 63.2%
unranked | z | 46.7%
unranked | t | 63.6%
unranked | r | 100.0%
s | p | 23.8%
s | z | 11.1%
s | t | 16.7%
d | p | 100.0%

 race+rank baseline accuracy: `62.5%`
