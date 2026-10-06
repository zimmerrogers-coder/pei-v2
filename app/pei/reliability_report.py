"""Runs the reliability analysis on the imported games, saves it to the
database, and draws reports/reliability.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.pei import build, config, plate_appearances, reliability, xstats  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"

_INK, _SECONDARY, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _BASELINE = "#2a78d6", "#c3c2b7"

LABELS = {
    "k_rate": "Strikeout rate", "bb_rate": "Walk rate", "hr_rate": "Home run rate",
    "woba": "wOBA", "xwoba": "xwOBA",
    "avg_ev": "Average exit speed", "hard_hit_rate": "Hard-hit rate",
    "sweet_spot_rate": "Sweet-spot rate", "barrel_rate": "Barrel rate",
    "xwobacon": "xwOBA on contact",
    "swing_rate": "Swing rate", "whiff_rate": "Whiff rate",
    "zone_swing_rate": "Zone-swing rate", "chase_rate": "Chase rate",
}


def draw(curve: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    metrics = list(summary["metric"])
    columns = 5
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 2.7 * rows), facecolor=_SURFACE,
                             sharey=True)
    for ax, metric in zip(axes.ravel(), metrics):
        info = summary.set_index("metric").loc[metric]
        points = curve[curve["metric"] == metric]
        reach = max(float(points["n"].max()) if len(points) else 10.0, 10.0) * 1.15

        ax.axhline(reliability.STABLE_R, color=_BASELINE, linewidth=1, linestyle=(0, (4, 3)))
        ax.axhline(0, color=_BASELINE, linewidth=1)
        if pd.notna(info["k"]):
            n = np.linspace(0, reach, 100)
            ax.plot(n, n / (n + info["k"]), color=_BLUE, linewidth=2, zorder=2)
            note = f"k = {info['k']:.0f}  (80% range {info['k_lo']:.0f}–" + (
                f"{info['k_hi']:.0f})" if info["k_hi"] < reliability.K_MAX * 0.99 else "no signal)")
        else:
            note = "no detectable signal"
        # dot area follows how many hitters are behind the point
        ax.scatter(points["n"], points["r"], s=10 + points["hitters"] * 1.2, color=_BLUE,
                   edgecolor=_SURFACE, linewidth=1, zorder=3)

        ax.set_title(LABELS[metric], loc="left", fontsize=10, color=_INK, pad=14)
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=7.5, color=_SECONDARY)
        ax.set_xlabel(info["denominator"], fontsize=8, color=_MUTED)
        ax.set_xlim(0, reach)
        ax.set_ylim(-0.35, 1.0)
        ax.set_yticks([0, 0.5, 0.7, 1.0])
        ax.tick_params(colors=_MUTED, labelsize=7.5, length=0)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor(_SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Reliability", fontsize=8, color=_MUTED)

    fig.suptitle("Split-half reliability by sample size.  Dots: measured (larger = more hitters).  "
                 "Line: fitted n / (n + k).  Dashed: 0.70.",
                 x=0.01, ha="left", fontsize=9.5, color=_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def compute(conn, splits: int = reliability.SPLITS, resamples: int = 50):
    settings = config.load()
    pitches = plate_appearances.load_pitches(conn)
    pa, _ = plate_appearances.derive(pitches, settings)
    pa["xwoba_con"] = xstats.out_of_fold(pa, settings)
    tables = reliability.event_tables(pa, pitches, settings)
    curve, summary = reliability.analyse(tables, splits=splits)
    summary = summary.merge(reliability.bootstrap_k(tables, resamples=resamples), on="metric")
    return curve, summary


def save(conn, curve: pd.DataFrame, summary: pd.DataFrame) -> None:
    conn.execute("TRUNCATE pei.reliability_curve, pei.metric_reliability")
    build._insert(conn, "pei.reliability_curve", curve, ["metric", "n", "hitters", "r_half", "r"])
    build._insert(conn, "pei.metric_reliability", summary, [
        "metric", "denominator", "league_value", "events", "hitters", "median_events",
        "max_events", "k", "k_lo", "k_hi", "share_no_signal", "stabilization_n",
        "largest_n_tested", "r_at_largest_n"])
    conn.commit()


def run(conn):
    curve, summary = compute(conn)
    save(conn, curve, summary)
    draw(curve, summary, REPORTS / "reliability.png")
    return curve, summary
