"""Scores the expected-stats model against v1's buckets and draws what it learned.

Writes reports/xstats_eval.json and reports/xstats_surface.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from app.pei import config, plate_appearances, xstats  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"

_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
_BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# outcome -> (label, colour, marker). Shape repeats what colour says.
_MARKS = {
    "OUT": ("Out", "#898781", "o"),
    "1B": ("Single", "#eb6834", "s"),
    "XB": ("Double or triple", "#1baf7a", "^"),
    "HR": ("Home run", "#4a3aa7", "D"),
}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=_INK, pad=10)
    ax.set_xlabel("Launch angle (degrees)", color=_MUTED, fontsize=9)
    ax.set_ylabel("Exit speed (mph)", color=_MUTED, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_surface(pa: pd.DataFrame, settings, path: Path) -> None:
    balls = xstats.measured(pa)
    model = xstats.SmoothModel().fit(balls, settings)

    # only draw the surface where there are balls to support it
    la_lo, la_hi = np.percentile(balls["launch_angle"], [1, 99])
    ev_lo, ev_hi = np.percentile(balls["exit_speed"], [1, 99])
    la, ev = np.meshgrid(np.linspace(la_lo, la_hi, 120), np.linspace(ev_lo, ev_hi, 120))
    grid = pd.DataFrame({"exit_speed": ev.ravel(), "launch_angle": la.ravel()})
    value = model.xwobacon(grid).reshape(ev.shape)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=_SURFACE, sharey=True)

    levels = np.linspace(0, 2.0, 11)
    filled = left.contourf(la, ev, np.clip(value, 0, 2.0), levels=levels, cmap=_BLUES)
    bar = fig.colorbar(filled, ax=left, pad=0.02, ticks=levels[::2])
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    bar.set_label("Expected wOBA on contact", color=_MUTED, fontsize=9)
    _style(left, "What the model learned")

    classes = xstats.contact_class(balls["outcome"])
    for key, (label, colour, marker) in _MARKS.items():
        chosen = balls[classes == key]
        right.scatter(chosen["launch_angle"], chosen["exit_speed"], s=26 if key == "OUT" else 38,
                      marker=marker, facecolor=colour, edgecolor=_SURFACE, linewidth=0.8,
                      alpha=0.55 if key == "OUT" else 0.95, label=f"{label} ({len(chosen)})",
                      zorder=2 if key == "OUT" else 3)
    right.set_xlim(la_lo, la_hi)
    right.set_ylim(ev_lo, ev_hi)
    right.grid(color=_GRID, linewidth=0.6)
    right.set_axisbelow(True)
    right.legend(frameon=False, fontsize=8, labelcolor=_INK, loc="lower right")
    _style(right, f"The {len(balls)} measured balls it learned from")
    right.set_ylabel("")

    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def run(conn, repeats: int = 4) -> pd.DataFrame:
    settings = config.load()
    pa, _ = plate_appearances.derive(plate_appearances.load_pitches(conn), settings)
    scores = xstats.evaluate(pa, settings, repeats=repeats)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "xstats_eval.json").write_text(
        json.dumps({"folds": 5, "repeats": repeats, "scores": scores.to_dict("records")}, indent=2),
        encoding="utf-8")
    draw_surface(pa, settings, REPORTS / "xstats_surface.png")
    return scores
