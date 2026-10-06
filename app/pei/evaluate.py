"""Does the model predict games it has not seen better than simpler methods do?

Every method gets the same deal: it may look only at the training games, and
it must predict each hitter's wOBA in the held-out games. Everything that is
learned from data (the expected-stats model, the shrinkage constants, the
league averages, the Bayesian model) is re-learned from the training games
alone for every split, so nothing from the held-out games can leak in.

Two ways of splitting the 12 games:

  chronological  train on the first 3, 6 or 9 games, predict the rest. This is
                 the real use: project the rest of the season from what has
                 happened so far.
  random halves  train on a random 6 games, predict the other 6. Not how the
                 tool is used, but it puts opposing hitters on both sides and
                 so tests far more hitters.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.pei import features, model, plate_appearances, reliability, xstats
from app.pei.config import Settings

BASELINES = ["league average", "raw wOBA to date", "shrunk wOBA", "shrunk xwOBA"]
MODEL = "PEI v2 model"
# Optional skill features: name of the variant -> {model component: [feature rates]}
FEATURE_SETS = {
    "model + whiff rate": {"k": ["whiff_rate"]},
    "model + chase rate": {"bb": ["chase_rate"]},
    "model + whiff and chase": {"k": ["whiff_rate"], "bb": ["chase_rate"]},
}
_FEATURE_EVENTS = {"whiff_rate": ("whiffs", "swings"), "chase_rate": ("chases", "chase_pitches")}


def chronological_splits(game_dates: dict, origins=(3, 6, 9)) -> list[dict]:
    ordered = sorted(game_dates, key=lambda g: (game_dates[g], g))
    return [{"scheme": "chronological", "split": f"first {n} games", "train": ordered[:n]}
            for n in origins if n < len(ordered)]


def random_half_splits(game_dates: dict, count: int = 8, seed: int = 0) -> list[dict]:
    """Random halves, always in mirrored pairs: train on half A and predict
    half B, then train on B and predict A. Scoring varies a lot from game to
    game (opposing pitching), so a one-directional split can make every method
    look biased just because its training half happened to be the tougher one.
    Using both directions cancels that."""
    games = sorted(game_dates)
    rng = np.random.default_rng(seed)
    splits = []
    for pair in range(count // 2):
        half = sorted(str(g) for g in rng.choice(games, size=len(games) // 2, replace=False))
        other = [g for g in games if g not in half]
        splits.append({"scheme": "random halves", "split": f"pair {pair + 1}, A to B", "train": half})
        splits.append({"scheme": "random halves", "split": f"pair {pair + 1}, B to A", "train": other})
    return splits


def training_window(pa: pd.DataFrame, pitches: pd.DataFrame, train_games, settings: Settings):
    """Everything derived from the training games only."""
    pa_train = pa[pa["game_id"].isin(train_games)].copy()
    pitches_train = pitches[pitches["game_id"].isin(train_games)]
    pa_train["xwoba_con"] = xstats.out_of_fold(pa_train, settings)

    totals = features.hitter_totals(
        features.hitter_game(pa_train, pitches_train, settings)).set_index("batter_id")
    tables = reliability.event_tables(pa_train, pitches_train, settings)
    wanted = ["woba", "xwoba", *_FEATURE_EVENTS]
    _, summary = reliability.analyse({m: tables[m] for m in wanted}, splits=100)
    summary = summary.set_index("metric")
    k = {m: (None if pd.isna(summary.loc[m, "k"]) else float(summary.loc[m, "k"])) for m in wanted}
    league = {m: float(summary.loc[m, "league_value"]) for m in wanted}
    return pa_train, totals, k, league


def baseline_predictions(totals: pd.DataFrame, k: dict, league: dict, hitters) -> pd.DataFrame:
    """Each baseline's predicted wOBA for every hitter asked about. A hitter
    with no training data gets the league average from every method."""
    seen = totals.reindex(hitters)
    pa = seen["pa"].fillna(0)
    out = pd.DataFrame(index=pd.Index(hitters, name="batter_id"))
    out["league average"] = league["woba"]
    out["raw wOBA to date"] = (seen["woba_num"] / pa.where(pa > 0)).fillna(league["woba"])
    out["shrunk wOBA"] = reliability.shrink(seen["woba_num"].fillna(0), pa, league["woba"], k["woba"])
    out["shrunk xwOBA"] = reliability.shrink(seen["xwoba_num"].fillna(0), pa, league["xwoba"],
                                             k["xwoba"])
    return out


def shrunk_features(totals: pd.DataFrame, k: dict, league: dict, feature_set: dict) -> dict:
    """Skill features for the model, each shrunk toward the league first so a
    hitter with three swings does not get an extreme value. A feature with no
    detectable signal in the training games is left out."""
    prepared = {}
    for component, names in feature_set.items():
        columns = {}
        for name in names:
            if k[name] is None:
                continue
            numerator, denominator = _FEATURE_EVENTS[name]
            columns[name] = reliability.shrink(totals[numerator], totals[denominator],
                                               league[name], k[name])
        if columns:
            prepared[component] = pd.DataFrame(columns, index=totals.index)
    return prepared


def model_predictions(pa_train: pd.DataFrame, settings: Settings, test_pa: pd.Series,
                      skill_features: dict | None = None, draws: int = 1500, tune: int = 1500,
                      seed: int = 0, n_sim: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Fit on the training games and predict each held-out hitter.

    Returns a table with `pred` (projected true-talent wOBA) and `lo`, `hi`:
    the 80% interval for the wOBA she will actually post over her held-out
    plate appearances. That interval is wider than the talent interval because
    it adds back the luck of a small number of trips to the plate."""
    data = model.prepare(pa_train, skill_features)
    _, trace = model.fit(data, settings, draws=draws, tune=tune, seed=seed)
    rng = np.random.default_rng(seed)
    w = settings.woba_weights

    keep = rng.choice(len(model._draws(trace, "p_hbp")), size=n_sim, replace=False)
    p_k, q_bb, contact = (model._draws(trace, n)[keep] for n in ("p_k", "q_bb", "contact"))
    p_hbp = model._draws(trace, "p_hbp")[keep]
    league = {n: model._draws(trace, n)[keep] for n in
              ("mu_k", "sigma_k", "mu_bb", "sigma_bb", "mu_c", "sigma_c")}
    row_of = {batter: row for row, batter in enumerate(data.batter_ids)}

    # what balls in play were actually worth in training: the luck to add back
    in_play = ~pa_train["outcome"].isin(["K", "BB", "HBP"])
    pool = pa_train.loc[in_play, "woba_value"].to_numpy(dtype=float)

    rows = []
    for batter, m in test_pa.items():
        m = int(m)
        if batter in row_of:
            pk, qbb, con = p_k[:, row_of[batter]], q_bb[:, row_of[batter]], contact[:, row_of[batter]]
        else:       # never seen: a new hitter drawn from the league distributions
            pk = 1 / (1 + np.exp(-(league["mu_k"] + league["sigma_k"] * rng.standard_normal(n_sim))))
            qbb = 1 / (1 + np.exp(-(league["mu_bb"] + league["sigma_bb"] * rng.standard_normal(n_sim))))
            con = np.exp(league["mu_c"] + league["sigma_c"] * rng.standard_normal(n_sim))
        ball_value = (1 - data.unmeasured_share) * con + data.unmeasured_share * data.unmeasured_value
        reach = 1 - pk
        talent = (reach * qbb * w["BB"] + reach * (1 - qbb) * p_hbp * w["HBP"]
                  + reach * (1 - qbb) * (1 - p_hbp) * ball_value)

        strikeouts = rng.binomial(m, pk)
        walks = rng.binomial(m - strikeouts, qbb)
        hit_by_pitch = rng.binomial(m - strikeouts - walks, p_hbp)
        balls = m - strikeouts - walks - hit_by_pitch
        # each simulated ball is a real training result, rescaled to her contact quality
        sampled = rng.choice(pool, size=(n_sim, m)) * (ball_value / pool.mean())[:, None]
        on_contact = (sampled * (np.arange(m)[None, :] < balls[:, None])).sum(axis=1)
        simulated = (walks * w["BB"] + hit_by_pitch * w["HBP"] + on_contact) / m
        low, high = np.percentile(simulated, model.INTERVAL)
        rows.append({"batter_id": batter, "pred": float(talent.mean()),
                     "lo": float(low), "hi": float(high)})
    return pd.DataFrame(rows).set_index("batter_id"), model.diagnostics(trace)


def evaluate_split(pa: pd.DataFrame, pitches: pd.DataFrame, split: dict, settings: Settings,
                   feature_sets: dict | None = None, seed: int = 0) -> tuple[pd.DataFrame, list]:
    """One row per held-out hitter: her actual wOBA and every method's prediction."""
    test = pa[~pa["game_id"].isin(split["train"])]
    actual = test.groupby("batter_id").agg(test_pa=("outcome", "size"),
                                           actual=("woba_value", "mean"))
    pa_train, totals, k, league = training_window(pa, pitches, split["train"], settings)

    rows = actual.join(baseline_predictions(totals, k, league, actual.index))
    rows["train_pa"] = totals["pa"].reindex(actual.index).fillna(0).astype(int)

    fits = []
    variants = {MODEL: {}, **(feature_sets or {})}
    for name, feature_set in variants.items():
        skill = shrunk_features(totals, k, league, feature_set)
        used = sorted(f for frame in skill.values() for f in frame.columns)
        predicted, diagnostics = model_predictions(pa_train, settings, actual["test_pa"], skill,
                                                   seed=seed)
        rows[name] = predicted["pred"]
        rows[f"{name} lo"], rows[f"{name} hi"] = predicted["lo"], predicted["hi"]
        fits.append({"scheme": split["scheme"], "split": split["split"], "method": name,
                     "features_used": used, **diagnostics})
    rows.insert(0, "split", split["split"])
    rows.insert(0, "scheme", split["scheme"])
    return rows.reset_index(), fits


def _weighted_rmse(rows: pd.DataFrame, method: str) -> float:
    error = rows[method] - rows["actual"]
    return float(np.sqrt(np.average(error ** 2, weights=rows["test_pa"])))


def score(rows: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Score each method on a set of held-out hitters. Errors are weighted by
    held-out plate appearances: missing on a hitter with 30 PA matters more
    than missing on one with 2, whose actual wOBA is mostly luck anyway."""
    out = []
    for method in methods:
        error = rows[method] - rows["actual"]
        entry = {
            "method": method, "hitters": len(rows), "holdout_pa": int(rows["test_pa"].sum()),
            "rmse": _weighted_rmse(rows, method),
            "mae": float(np.average(error.abs(), weights=rows["test_pa"])),
            "bias": float(np.average(error, weights=rows["test_pa"])),
            "rank_corr": (float(spearmanr(rows[method], rows["actual"]).statistic)
                          if rows[method].nunique() > 1 else None),
            "coverage_80": None,
        }
        if f"{method} lo" in rows:
            inside = (rows[f"{method} lo"] <= rows["actual"]) & (rows["actual"] <= rows[f"{method} hi"])
            entry["coverage_80"] = float(inside.mean())
        out.append(entry)
    return pd.DataFrame(out)


def luck_floor(rows: pd.DataFrame, per_pa_sd: float) -> float:
    """The error a perfect forecaster would still make. Even if a hitter's true
    talent were known exactly, her wOBA over m plate appearances would miss it
    by about (spread of one PA) / sqrt(m). This is that, weighted the same way
    as the scores. No method can be expected to beat it."""
    return float(np.sqrt(np.average(per_pa_sd ** 2 / rows["test_pa"], weights=rows["test_pa"])))


def compare(rows: pd.DataFrame, method: str, other: str, resamples: int = 4000,
            seed: int = 0) -> dict:
    """How sure is it that `method` beats `other`? Redraw the held-out hitters
    with replacement many times and see how often its error is the smaller one."""
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        sample = rows.iloc[rng.integers(0, len(rows), size=len(rows))]
        differences.append(_weighted_rmse(sample, other) - _weighted_rmse(sample, method))
    differences = np.array(differences)
    low, high = np.percentile(differences, [10, 90])
    return {"method": method, "versus": other,
            "rmse_gain": _weighted_rmse(rows, other) - _weighted_rmse(rows, method),
            "gain_lo": float(low), "gain_hi": float(high),
            "share_better": float((differences > 0).mean())}
