# PEI v2: Workflow, Schema and Build Guide

Revised after the build. This describes what was built and how each step is
checked. Results from the program's own games are not included in this repository.

## 1. Goal

Turn PEI from a descriptive index into a **projection**: estimate each hitter's
true-talent wOBA, with an honest range around it, from pitch-tracking data, and
test that estimate on games the model did not see.

| v1 problem | v2 fix |
|---|---|
| One tournament file, median 5 PA, no way to add games | Per-game TrackMan and Dartfish files imported into PostgreSQL; duplicates refused |
| Placeholder metrics set to 0 (chase, zone-swing, whiff, baserunning) | Discipline rates computed from pitch location; baserunning dropped because no file credits a runner |
| xwOBA set equal to wOBA; hand-set bucket values | Fitted exit-speed/launch-angle model, scored on held-out balls |
| Weights chosen by hand (40/25/15/10/10/5) | A Bayesian model whose shrinkage is learned; candidate skill features kept only if they improve holdout error |
| Raw small samples taken at face value | Split-half reliability per metric; partial pooling toward the league |
| No uncertainty | 80% interval on every projection, checked for calibration |
| No versioning | Every run stored with settings, diagnostics and holdout scores |

## 2. Folder layout

```
app/
  db.py                    connect; run sql/NN_*.sql in order
  importers/
    trackman.py            read a TrackMan file into clean values
    dartfish.py            read a Dartfish file; line it up with a game; list conflicts
    store.py               write to core.*; refuse duplicates; delete an import
  pei/
    config.py              load config/pei.yaml
    plate_appearances.py   pitches -> plate appearances and outcomes
    features.py            per-game counts; rates as sum over sum
    validate.py            nine consistency checks
    xstats.py              EV/LA model -> xwOBAcon; comparison with v1 buckets
    reliability.py         split-half, Spearman-Brown, k, bootstrap, shrinkage
    model.py               the PyMC model; simulation of known-talent hitters
    fit.py                 fit, diagnostics, predictive checks, recovery test, save
    evaluate.py            train/holdout splits, baselines, scoring
    *_report.py            charts and saved results for each stage
    __main__.py            command line
config/pei.yaml            wOBA weights, batted-ball thresholds, strike zone
sql/01..06_*.sql           table definitions
scripts/create_database.py one-time role and database setup
tests/                     46 pytest checks
reports/                   charts and saved results, written when the pipeline runs (not committed)
```

## 3. PostgreSQL tables

Two schemas in one database. `core` is data as imported; `pei` is everything
derived from it and can be rebuilt at any time.

| Table | One row is | Key |
|---|---|---|
| `core.import_file` | an imported file, with a content hash | `import_id`; unique (source, hash) |
| `core.team` | a team (TrackMan code, readable name) | `team_code` |
| `core.player` | a player on a team | `player_id`; unique (name, team) |
| `core.game` | a game | TrackMan `GameID` |
| `core.pitch` | a TrackMan pitch: typed columns plus the full row as JSONB | `PitchUID` |
| `core.dartfish_pitch` | the video tags for a pitch | `PitchUID` |
| `pei.plate_appearance` | a completed PA: outcome, wOBA value, contact flags, xwOBAcon | unique (game, inning, half, PA number) |
| `pei.hitter_game` | counts for a hitter in a game | (game, hitter) |
| `pei.reliability_curve` | reliability of a metric at a sample size | (metric, n) |
| `pei.metric_reliability` | a metric's k, its range, its 0.70 point | metric |
| `pei.model_version` | a model run: data used, settings, diagnostics, checks, holdout RMSE | version |
| `pei.model_parameter` | a league-level parameter with its 80% interval | (version, parameter) |
| `pei.projection` | a hitter's projection: wOBA, K rate, BB rate, contact, each with an interval | (version, hitter) |
| `pei.evaluation` | a method's holdout score under a split scheme | (version, scheme, method) |
| `pei.evaluation_comparison` | the model against one other method | (version, scheme, method, versus) |

Players are keyed on name and team because the files' `BatterId` is not a
reliable player ID.

## 4. Run order, with the check after each step

| Step | What happens | Check |
|---|---|---|
| 1 Import | TrackMan files into `core`; Dartfish joined by pitch order, verified by matching release speeds | Row counts match the files |
| 2 Resolve players | "Last, First" and "22 First Last" stored one way | No duplicate players |
| 3 Plate appearances | Group pitches; map outcomes; error = reached on error; sacrifices split into bunt and fly | Nine consistency checks |
| 4 Features | Counts per hitter-game; every rate per PA | Every hitter recounted from the raw files by separate code |
| 5 Expected stats | EV/LA model for xwOBAcon, out-of-fold by game | Holdout log loss beats v1 buckets |
| 6 Reliability | Split-half over 200 splits; fit k; bootstrap its range | Saved to `pei.metric_reliability` |
| 7 Model | Hierarchical PyMC model on all games | No divergences; r-hat < 1.01 |
| 8 Evaluate | Train on early games, predict the rest; all components refitted per split | RMSE beats every baseline |
| 9 Log | Projections, parameters, diagnostics and scores stored by version | A version can be compared with the last |

## 5. Key methods

**Split-half reliability.** For a metric at sample size n: take each hitter with
at least n events, deal n of them at random into two halves, compute the metric
in each, and correlate the halves across hitters. Average over 200 random
splits, then step up to the full n:

```
r_n = 2r / (1 + r)                      Spearman-Brown
r_n = n / (n + k)                       fitted across sample sizes
```

k is the sample size at which the metric is half signal. The 0.70 point is
`0.7/0.3 x k`.

**Shrinkage** (the baseline the model is compared with):

```
shrunk = (hitter's count + k x league rate) / (hitter's events + k)
```

The constant is k, not the 0.70 point. The earlier version of this guide named
the 0.70 point, which would shrink 2.33 times too hard.

**Hierarchical projection (PyMC).** A plate appearance is a sequence of steps,
and each hitter-level skill is drawn from a league distribution:

```python
with pm.Model(coords={"hitter": ids}):
    def skill(name, centre, centre_sd, spread_sd):
        mu    = pm.Normal(f"mu_{name}", centre, centre_sd)        # league centre
        sigma = pm.HalfNormal(f"sigma_{name}", spread_sd)         # spread of true talent
        z     = pm.Normal(f"z_{name}", 0, 1, dims="hitter")
        return mu + sigma * z                                     # non-centred

    p_k  = pm.math.invlogit(skill("k",  logit(0.15), 1.0, 1.0))   # strikeout
    q_bb = pm.math.invlogit(skill("bb", logit(0.15), 1.0, 1.0))   # walk, given no strikeout
    contact = pm.math.exp(skill("c", np.log(0.37), 0.5, 0.5))     # mean xwOBAcon
    p_hbp = pm.Beta("p_hbp", 1, 30)                               # league-level
    alpha = pm.Gamma("alpha", 2, 1)

    pm.Binomial("k_obs",  n=pa,     p=p_k,  observed=k)
    pm.Binomial("bb_obs", n=pa - k, p=q_bb, observed=bb)
    pm.Binomial("hbp_obs", n=(pa - k - bb).sum(), p=p_hbp, observed=hbp.sum())
    pm.Gamma("ball_obs", alpha=alpha, beta=alpha / contact[ball_hitter], observed=ball_value)

    reach = 1 - p_k
    woba = pm.Deterministic("woba",
        reach * q_bb * 0.69 + reach * (1 - q_bb) * p_hbp * 0.72
        + reach * (1 - q_bb) * (1 - p_hbp) * contact)
    trace = pm.sample(1500, tune=1500, chains=4, target_accept=0.95)
```

The posterior for `woba` gives each hitter's projection and 80% interval. The
earlier plan's Normal likelihood on per-game wOBA was replaced because a
three-PA game is not close to Normal and its noise term was never defined.

## 6. How to evaluate

- **No leakage.** Refit the expected-stats model, league averages, shrinkage
  constants and the Bayesian model on training games only, for every split.
- **Rolling origin.** Train through game 3, 6 and 9; predict the rest. Score
  only hitters seen in training.
- **Random halves in mirrored pairs** as a second view with more hitters. A
  one-directional split was biased by game-to-game differences in opposing
  pitching.
- **Baselines.** League average, raw wOBA to date, shrunk wOBA, shrunk xwOBA.
- **Report** PA-weighted RMSE, MAE, bias, rank correlation, coverage of the 80%
  predictive interval, and the error luck alone would produce.
- **State the uncertainty of the comparison.** Resample held-out hitters and
  report how often the model's error is the smaller one.
- **Keep a feature only if it lowers holdout error.**
- **Before any of that, test on simulated hitters** whose true talent is known.

## 7. Phases

| Phase | Work | Status |
|---|---|---|
| A Setup | Virtual environment, database, schemas, PyMC check | Done |
| B Import | TrackMan and Dartfish readers, join, duplicate refusal | Done |
| C Plate appearances and validation | Outcomes, counts, nine checks, hand recount | Done |
| D Expected stats | EV/LA model against v1 buckets | Done |
| E Reliability | Split-half, k, bootstrap, shrinkage | Done |
| F Model and evaluation | PyMC model, recovery test, holdout test | Done |
| G Report | Model card, this guide, README | Done |
| H Desktop tab | Leaderboard with intervals, player card, model checks, reliability chart; anonymised export | Not started |

## 8. What counts as done

| Criterion |
|---|
| One command runs import to projections from an empty database |
| All validation checks pass or are explained |
| The fit is clean: no divergences, r-hat under 1.01 |
| The model recovers known talent on simulated data |
| The model is compared with every baseline on holdout RMSE, with the uncertainty of that comparison |
| Intervals are calibrated on held-out games |
| Coach-facing score and desktop tab |
