"""Runs the full holdout evaluation, saves it, and draws reports/evaluation.png.

About 20 minutes: the model is refitted for every split.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.pei import build, config, evaluate, plate_appearances  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"

_INK, _SECONDARY, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _NEUTRAL = "#2a78d6", "#898781"

_SCORE_COLUMNS = ["method", "hitters", "holdout_pa", "rmse", "mae", "bias", "rank_corr",
                  "coverage_80"]


def run(conn, version: str, random_halves: int = 8) -> dict:
    settings = config.load()
    pitches = plate_appearances.load_pitches(conn)
    pa, _ = plate_appearances.derive(pitches, settings)
    dates = dict(conn.execute("SELECT game_id, game_date FROM core.game").fetchall())

    splits = [(s, evaluate.FEATURE_SETS) for s in evaluate.chronological_splits(dates)]
    splits += [(s, None) for s in evaluate.random_half_splits(dates, count=random_halves)]

    all_rows, fits = [], []
    for number, (split, feature_sets) in enumerate(splits):
        rows, split_fits = evaluate.evaluate_split(pa, pitches, split, settings, feature_sets,
                                                   seed=number)
        all_rows.append(rows)
        fits.extend(split_fits)
        print(f"  done: {split['scheme']}, {split['split']}", flush=True)
    rows = pd.concat(all_rows, ignore_index=True)

    per_pa_sd = float(pa["woba_value"].std())
    scores, comparisons, floors = [], [], {}
    for scheme, group in rows.groupby("scheme"):
        # only hitters the methods had something to go on; for anyone unseen
        # every method says "league average" and there is nothing to compare
        seen = group[group["train_pa"] > 0]
        methods = evaluate.BASELINES + [evaluate.MODEL]
        if scheme == "chronological":
            methods += list(evaluate.FEATURE_SETS)
        scores.append(evaluate.score(seen, methods).assign(scheme=scheme))
        floors[scheme] = evaluate.luck_floor(seen, per_pa_sd)
        for other in [m for m in methods if m != evaluate.MODEL]:
            comparisons.append({"scheme": scheme, **evaluate.compare(seen, evaluate.MODEL, other)})
    scores = pd.concat(scores, ignore_index=True)
    comparisons = pd.DataFrame(comparisons)

    _save(conn, version, scores, comparisons)
    result = {"version": version, "scores": scores, "comparisons": comparisons, "fits": fits,
              "rows": rows, "luck_floor": floors}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.json").write_text(json.dumps({
        "version": version,
        "luck_floor": floors,
        "scores": scores.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "fits": fits,
    }, indent=2, default=str), encoding="utf-8")
    draw(scores, floors, REPORTS / "evaluation.png")
    return result


def _save(conn, version: str, scores: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    conn.execute("DELETE FROM pei.evaluation WHERE version = %s", (version,))
    conn.execute("DELETE FROM pei.evaluation_comparison WHERE version = %s", (version,))
    build._insert(conn, "pei.evaluation", scores.assign(version=version),
                  ["version", "scheme", *_SCORE_COLUMNS])
    build._insert(conn, "pei.evaluation_comparison", comparisons.assign(version=version),
                  ["version", "scheme", "method", "versus", "rmse_gain", "gain_lo", "gain_hi",
                   "share_better"])
    chrono = scores[scores["scheme"] == "chronological"].set_index("method")
    baselines = chrono.loc[evaluate.BASELINES, "rmse"]
    conn.execute(
        "UPDATE pei.model_version SET holdout_rmse = %s, best_baseline = %s, "
        "best_baseline_rmse = %s WHERE version = %s",
        (float(chrono.loc[evaluate.MODEL, "rmse"]), baselines.idxmin(), float(baselines.min()),
         version))
    conn.commit()


def draw(scores: pd.DataFrame, floors: dict, path: Path) -> None:
    schemes = ["chronological", "random halves"]
    titles = {"chronological": "Train on the first 3, 6 or 9 games, predict the rest",
              "random halves": "Train on a random 6 games, predict the other 6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=_SURFACE)
    for ax, scheme in zip(axes, schemes):
        part = scores[scores["scheme"] == scheme].sort_values("rmse", ascending=False)
        colours = [_BLUE if m == evaluate.MODEL else _NEUTRAL for m in part["method"]]
        floor = floors[scheme]
        low = min(part["rmse"].min(), floor)
        ax.axvline(floor, color=_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("luck alone", (floor, len(part) - 0.45), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=_SECONDARY, va="center")
        ax.hlines(part["method"], low - 0.004, part["rmse"], color=_GRID, linewidth=1.5, zorder=1)
        ax.scatter(part["rmse"], part["method"], s=70, color=colours, edgecolor=_SURFACE,
                   linewidth=1.5, zorder=3)
        for value, method in zip(part["rmse"], part["method"]):
            ax.annotate(f"{value:.3f}", (value, method), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=_INK)
        first = part.iloc[0]
        ax.set_title(titles[scheme], loc="left", fontsize=10.5, color=_INK, pad=16)
        ax.text(0, 1.02, f"{first['hitters']} hitter-splits, {first['holdout_pa']:,} held-out "
                         "plate appearances", transform=ax.transAxes, fontsize=8, color=_SECONDARY)
        ax.set_xlabel("Error predicting held-out wOBA (RMSE, lower is better)",
                      fontsize=8.5, color=_MUTED)
        ax.set_xlim(low - 0.004, part["rmse"].max() + 0.006)
        ax.set_ylim(-0.6, len(part) - 0.1)
        ax.tick_params(axis="x", colors=_MUTED, labelsize=8, length=0)
        ax.tick_params(axis="y", colors=_INK, labelsize=9, length=0)
        ax.set_facecolor(_SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(w_pad=3)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def redraw() -> None:
    """Draw the chart again from the saved results, without refitting anything."""
    saved = json.loads((REPORTS / "evaluation.json").read_text(encoding="utf-8"))
    draw(pd.DataFrame(saved["scores"]), saved["luck_floor"], REPORTS / "evaluation.png")
