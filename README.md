# PEI v2: Bayesian Hitter Projections

## Overview:

PEI v1 was a single script that ranked hitters from one tournament using weights I chose by hand. For v2 I wanted to answer a different question: what is each hitter's true-talent wOBA, and how confident can we be in that number? This project imports pitch-tracking files into PostgreSQL, builds plate appearances from them, and fits a hierarchical Bayesian model in PyMC that gives every hitter a projected wOBA with an 80% interval. The model is tested on games it has not seen against simpler baselines.

This repository is the code only. The game data and the results from it belong to the program I work for, so they are not included.

## How It Works:

A plate appearance is treated as a sequence, where the hitter either strikes out, walks, gets hit by a pitch, or puts the ball in play, which is worth her contact quality. Contact quality comes from an expected-wOBA model on exit speed and launch angle. Each skill is partially pooled across hitters, so the model learns how far to pull a small sample toward the league instead of me setting that amount by hand.

Before the model runs, every build goes through nine consistency checks, such as making sure PA = AB + BB + HBP + SH + SF for every hitter in every game. The evaluation refits every piece that learns from the data, including the expected-stats model and the league averages, on the training games only, so nothing from the holdout games leaks into the projection. The model is also tested on simulated hitters whose true talent is known before it is trusted on real ones.

## Running It:

This needs Python 3.13 and a local PostgreSQL service. One-time setup:

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\create_database.py
```

`create_database.py` asks for the `postgres` admin password once, creates the `metrics_app` role and the `advanced_metrics` and `advanced_metrics_test` databases, and writes the app's password to `pgpass.conf`. No password is stored in this project.

From a folder of TrackMan and Dartfish game files to projections:

```
.venv\Scripts\python.exe -m app.pei all "C:\path\to\game files"
```

The steps can also be run one at a time:

| Command | What it does |
|---|---|
| `python -m app.importers <folder or files>` | Imports TrackMan files, then Dartfish files |
| `python -m app.pei build` | Builds plate appearances and features and runs the nine checks |
| `python -m app.pei xstats` | Scores the expected-stats model |
| `python -m app.pei reliability` | Split-half reliability for each metric |
| `python -m app.pei fit --version v2.0` | Fits the model and saves the projections |
| `python -m app.pei recovery` | Tests the model on simulated hitters |
| `python -m app.pei evaluate` | Holdout test against the baselines |
| `python -m pytest` | Runs the tests |

The tests that need real game files run only when `PEI_DATA_DIR` points at a folder of them. The rest build their own small files.

## Layout:

```
app/importers/   read TrackMan and Dartfish files and write them to the database
app/pei/         plate appearances, features, expected stats, reliability, model, evaluation
config/          wOBA weights and thresholds
sql/             table definitions, run in name order by app/db.py
scripts/         one-time setup
tests/           pytest checks
docs/            workflow and schema guide
```

## Data:

Raw game files are never committed, and neither is anything the pipeline writes to `reports/`. Only TrackMan and Dartfish files are imported, and Yakkertech files are skipped and listed. The same file cannot be imported twice, and a file containing pitches already in the database is refused.
