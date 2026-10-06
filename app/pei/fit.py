"""Runs the model: fit it on the imported games, check it, save the projections.

Also holds the two checks that say whether a fit deserves to be believed:

  predictive_checks  does data simulated from the fitted model look like the real data?
  recovery           given invented hitters with known talent, does the model find it?
"""

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from app.pei import build, config, model, plate_appearances, xstats  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"
MAX_RHAT, MIN_ESS = 1.01, 400

_INK, _SECONDARY, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _BASELINE = "#2a78d6", "#c3c2b7"

_PROJECTION_COLUMNS = [
    "batter_id", "n_pa", "n_balls", "raw_woba", "proj_woba", "woba_lo", "woba_hi",
    "raw_k_rate", "proj_k_rate", "k_rate_lo", "k_rate_hi",
    "raw_bb_rate", "proj_bb_rate", "bb_rate_lo", "bb_rate_hi",
    "proj_contact", "contact_lo", "contact_hi"]


def fit_is_clean(diagnostics: dict) -> bool:
    return (diagnostics["divergences"] == 0 and diagnostics["max_rhat"] < MAX_RHAT
            and diagnostics["min_ess"] > MIN_ESS)


def predictive_checks(fitted_model, trace, data: model.ModelData, seed: int = 1) -> list[dict]:
    """Simulate whole new datasets from the fitted model and compare a few
    summaries with the real one. If the real value sits far outside what the
    model produces, the model is wrong about that feature of the data."""
    with fitted_model:
        simulated = pm.sample_posterior_predictive(trace, random_seed=seed, progressbar=False)

    def draws(name):
        values = np.asarray(simulated.posterior_predictive[name].values)
        return values.reshape(-1, *values.shape[2:])

    k, bb, balls = draws("k_obs"), draws("bb_obs"), draws("ball_obs")
    comparisons = [
        ("total strikeouts", data.k.sum(), k.sum(axis=1)),
        ("most strikeouts by one hitter", data.k.max(), k.max(axis=1)),
        ("hitters with no strikeouts", (data.k == 0).sum(), (k == 0).sum(axis=1)),
        ("total walks", data.bb.sum(), bb.sum(axis=1)),
        ("average batted-ball value", data.ball_value.mean(), balls.mean(axis=1)),
        ("spread of batted-ball values", data.ball_value.std(), balls.std(axis=1)),
        ("share of balls worth over 1.0", (data.ball_value > 1).mean(), (balls > 1).mean(axis=1)),
    ]
    results = []
    for name, observed, replicated in comparisons:
        low, high = np.percentile(replicated, [5, 95])
        results.append({"check": name, "observed": float(observed), "model_lo": float(low),
                        "model_hi": float(high), "inside": bool(low <= observed <= high)})
    return results


def run(conn, version: str, notes: str = "", draws: int = 1500, tune: int = 1500) -> dict:
    """Fit on every imported game and store the result under `version`,
    replacing any earlier run of the same name."""
    settings = config.load()
    pitches = plate_appearances.load_pitches(conn)
    pa, _ = plate_appearances.derive(pitches, settings)
    pa["xwoba_con"] = xstats.out_of_fold(pa, settings)
    data = model.prepare(pa)

    fitted_model, trace = model.fit(data, settings, draws=draws, tune=tune)
    diagnostics = model.diagnostics(trace)
    checks = predictive_checks(fitted_model, trace, data)
    hitters = model.hitter_summary(trace, data)
    league = model.league_summary(trace, data)

    game_dates = dict(conn.execute("SELECT game_id, game_date FROM core.game").fetchall())
    used = pa["game_id"].unique()
    conn.execute("DELETE FROM pei.model_version WHERE version = %s", (version,))
    conn.execute(
        "INSERT INTO pei.model_version (version, games, train_through, hitters, "
        "plate_appearances, measured_balls, settings, diagnostics, checks, notes) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (version, len(used), max(game_dates[g] for g in used), data.n_hitters, int(data.pa.sum()),
         len(data.ball_value), Jsonb(dataclasses.asdict(settings)), Jsonb(diagnostics),
         Jsonb(checks), notes))
    build._insert(conn, "pei.projection", hitters.assign(version=version),
                  ["version", *_PROJECTION_COLUMNS])
    build._insert(conn, "pei.model_parameter", league.assign(version=version),
                  ["version", "parameter", "mean", "lo_80", "hi_80"])
    conn.commit()
    return {"version": version, "diagnostics": diagnostics, "checks": checks,
            "hitters": hitters, "league": league, "clean": fit_is_clean(diagnostics)}


def recovery(seed: int = 11, n_hitters: int = 66, draws: int = 1500, tune: int = 1500) -> dict:
    """Fit the model to invented hitters and compare with the truth it was
    invented from. Reports how close the projections are, how close the raw
    numbers are, and how often the 80% interval contains the true value."""
    settings = config.load()
    pa, truth = model.simulate(n_hitters=n_hitters, seed=seed, settings=settings)
    data = model.prepare(pa)
    _, trace = model.fit(data, settings, draws=draws, tune=tune, seed=seed)
    table = model.hitter_summary(trace, data).merge(
        truth.drop(columns="n_pa"), on="batter_id")

    def rmse(column):
        return float(np.sqrt(np.mean((table[column] - table["true_woba"]) ** 2)))

    inside = (table["woba_lo"] <= table["true_woba"]) & (table["true_woba"] <= table["woba_hi"])
    return {"table": table, "diagnostics": model.diagnostics(trace),
            "rmse_raw": rmse("raw_woba"), "rmse_model": rmse("proj_woba"),
            "coverage_80": float(inside.mean()), "hitters": len(table), "seed": seed}


def draw_recovery(result: dict, path: Path) -> None:
    table = result["table"]
    lo = 0.0
    hi = float(max(table["raw_woba"].max(), table["woba_hi"].max(), table["true_woba"].max())) * 1.05
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=_SURFACE, sharey=True)

    panels = [
        (left, "raw_woba", f"Raw wOBA: off by {result['rmse_raw']:.3f} on average"),
        (right, "proj_woba", f"Model projection: off by {result['rmse_model']:.3f} on average"),
    ]
    for ax, column, title in panels:
        ax.plot([lo, hi], [lo, hi], color=_BASELINE, linewidth=1, zorder=1)
        if column == "proj_woba":
            ax.vlines(table["true_woba"], table["woba_lo"], table["woba_hi"], color=_BLUE,
                      linewidth=1, alpha=0.35, zorder=2)
        ax.scatter(table["true_woba"], table[column], s=10 + table["n_pa"] * 1.6, color=_BLUE,
                   edgecolor=_SURFACE, linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=11, color=_INK, pad=10)
        ax.set_xlabel("True wOBA (known, because the hitters are simulated)",
                      fontsize=9, color=_MUTED)
        ax.set_xlim(0.15, 0.65)
        ax.set_ylim(lo, hi)
        ax.grid(color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_MUTED, labelsize=8, length=0)
        ax.set_facecolor(_SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
    left.set_ylabel("Estimate", fontsize=9, color=_MUTED)
    fig.suptitle(
        f"{result['hitters']} simulated hitters. Larger dots have more plate appearances. "
        f"A dot on the line is a perfect estimate. Bars are 80% intervals; "
        f"{result['coverage_80']:.0%} contain the truth.",
        x=0.01, ha="left", fontsize=9.5, color=_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def run_recovery(seeds=(11, 1, 2, 3, 4)) -> dict:
    """Run the recovery test on several independently invented leagues. One
    league of 66 hitters is a small sample, so its interval coverage bounces
    around; the average over several is the number to read. The chart shows
    the first league."""
    results = [recovery(seed=seed) for seed in seeds]
    draw_recovery(results[0], REPORTS / "model_recovery.png")
    keys = ("seed", "hitters", "rmse_raw", "rmse_model", "coverage_80", "diagnostics")
    runs = [{k: r[k] for k in keys} for r in results]
    average = {k: float(np.mean([r[k] for r in runs]))
               for k in ("rmse_raw", "rmse_model", "coverage_80")}
    (REPORTS / "model_recovery.json").write_text(
        json.dumps({"average": average, "runs": runs}, indent=2), encoding="utf-8")
    return {"runs": runs, "average": average}
