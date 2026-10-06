"""How many plate appearances (or batted balls, or swings) before a number means something?

Split-half reliability answers that. Take n of a hitter's plate appearances,
deal them at random into two halves, and compute the rate in each half. Do it
for every hitter and correlate the two columns. If the metric measures a real
skill, hitters who are high in one half are high in the other. If it is mostly
luck, the halves do not agree.

Each half holds only n/2 events, so the Spearman-Brown formula steps the
correlation up to what n events would give:  r_n = 2r / (1 + r).

Reliability follows a known curve as the sample grows:

    r_n = n / (n + k)

k is the one number that describes the metric. It is the sample size at which
the metric is half signal and half noise (r = 0.5), and it is exactly the
constant the shrinkage formula needs. The "stabilization point" people quote,
where r reaches 0.70, is 0.7/0.3 x k, or about 2.33 k.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.pei import features, plate_appearances
from app.pei.config import Settings

SPLITS = 200
MIN_HITTERS = 10          # a correlation across fewer hitters than this is not reported
STABLE_R = 0.70
K_MAX = 100_000.0         # a fitted k at this ceiling means "no detectable signal"

# metric -> what one "event" is
DENOMINATOR = {
    "k_rate": "PA", "bb_rate": "PA", "hr_rate": "PA", "woba": "PA", "xwoba": "PA",
    "avg_ev": "measured balls", "hard_hit_rate": "measured balls",
    "sweet_spot_rate": "measured balls", "barrel_rate": "measured balls",
    "xwobacon": "measured balls",
    "swing_rate": "pitches", "whiff_rate": "swings",
    "zone_swing_rate": "pitches in zone", "chase_rate": "pitches out of zone",
}


def event_tables(pa: pd.DataFrame, pitches: pd.DataFrame, settings: Settings) -> dict:
    """One table per metric: a row per event with the hitter and the event's
    value (1/0 for a rate, the number itself for exit speed or wOBA)."""
    def table(frame, values):
        return pd.DataFrame({"batter_id": frame["batter_id"].to_numpy(),
                             "game_id": frame["game_id"].to_numpy(),
                             "value": np.asarray(values, dtype=float)})

    balls = pa[pa["hard_hit"].notna()]
    kept = pitches.merge(pa[plate_appearances.PA_KEY], how="inner")
    f = features.pitch_flags(kept, settings)
    swings, zone = f[f["swing"]], f[f["in_zone"]]
    outside = f[f["located"] & ~f["in_zone"]]
    return {
        "k_rate": table(pa, pa["outcome"] == "K"),
        "bb_rate": table(pa, pa["outcome"] == "BB"),
        "hr_rate": table(pa, pa["outcome"] == "HR"),
        "woba": table(pa, pa["woba_value"]),
        "xwoba": table(pa, pa["xwoba_con"].fillna(pa["woba_value"])),
        "avg_ev": table(balls, balls["exit_speed"]),
        "hard_hit_rate": table(balls, balls["hard_hit"].astype(bool)),
        "sweet_spot_rate": table(balls, balls["sweet_spot"].astype(bool)),
        "barrel_rate": table(balls, balls["barrel"].astype(bool)),
        "xwobacon": table(balls, balls["xwoba_con"]),
        "swing_rate": table(f, f["swing"]),
        "whiff_rate": table(swings, swings["whiff"]),
        "zone_swing_rate": table(zone, zone["swing"]),
        "chase_rate": table(outside, outside["swing"]),
    }


def spearman_brown(r_half: float) -> float:
    return 2 * r_half / (1 + r_half)


def split_half(events: pd.DataFrame, n: int, splits: int = SPLITS, seed: int = 0) -> dict | None:
    """Reliability of a metric at sample size n (n must be even).
    Returns None when fewer than MIN_HITTERS hitters have n events."""
    values = [g["value"].to_numpy() for _, g in events.groupby("batter_id") if len(g) >= n]
    if len(values) < MIN_HITTERS:
        return None
    rng = np.random.default_rng(seed)
    half = n // 2
    correlations = []
    for _ in range(splits):
        a, b = np.empty(len(values)), np.empty(len(values))
        for i, v in enumerate(values):
            drawn = rng.choice(v, size=2 * half, replace=False)
            a[i], b[i] = drawn[:half].mean(), drawn[half:].mean()
        if a.std() > 0 and b.std() > 0:        # a half where every hitter ties has no correlation
            correlations.append(np.corrcoef(a, b)[0, 1])
    if not correlations:
        return None
    r_half = float(np.mean(correlations))
    return {"n": 2 * half, "hitters": len(values), "r_half": r_half, "r": spearman_brown(r_half)}


def sample_sizes(events: pd.DataFrame) -> list[int]:
    """Even sample sizes to test: from 4 up to the largest n that at least
    MIN_HITTERS hitters reach, at most a dozen of them."""
    counts = np.sort(events.groupby("batter_id").size().to_numpy())[::-1]
    if len(counts) < MIN_HITTERS:
        return []
    largest = int(counts[MIN_HITTERS - 1]) // 2 * 2
    if largest < 4:
        return []
    return sorted({int(x) // 2 * 2 for x in np.linspace(4, largest, 12)})


def fit_k(curve: pd.DataFrame) -> float:
    """The k that makes n / (n + k) pass closest to the measured points. Points
    backed by more hitters count for more. Returns K_MAX when the points show
    no rising signal."""
    n, r, weight = (curve[c].to_numpy(dtype=float) for c in ("n", "r", "hitters"))

    def miss(log_k):
        return float(np.sum(weight * (r - n / (n + np.exp(log_k))) ** 2))

    best = minimize_scalar(miss, bounds=(np.log(0.5), np.log(K_MAX)), method="bounded")
    return float(np.exp(best.x))


def analyse(tables: dict, splits: int = SPLITS, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (curve, summary).
    curve:   one row per metric and sample size, with the measured reliability
    summary: one row per metric, with k, the stabilization point and the league value"""
    curve_rows, summary_rows = [], []
    for metric, events in tables.items():
        points = [p for n in sample_sizes(events)
                  if (p := split_half(events, n, splits, seed)) is not None]
        curve = pd.DataFrame(points, columns=["n", "hitters", "r_half", "r"])
        curve.insert(0, "metric", metric)
        curve_rows.append(curve)

        k = fit_k(curve) if len(curve) >= 2 else K_MAX
        detectable = k < K_MAX * 0.99
        per_hitter = events.groupby("batter_id").size()
        summary_rows.append({
            "metric": metric,
            "denominator": DENOMINATOR[metric],
            "league_value": float(events["value"].mean()),
            "events": len(events),
            "hitters": int(len(per_hitter)),
            "median_events": float(per_hitter.median()),
            "max_events": int(per_hitter.max()),
            "k": k if detectable else None,
            "stabilization_n": k * STABLE_R / (1 - STABLE_R) if detectable else None,
            "largest_n_tested": int(curve["n"].max()) if len(curve) else None,
            "r_at_largest_n": float(curve["r"].iloc[-1]) if len(curve) else None,
        })
    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(summary_rows)


def bootstrap_k(tables: dict, resamples: int = 50, splits: int = 20, seed: int = 1) -> pd.DataFrame:
    """How sure are we of each k? Redo the whole analysis many times, each time
    on a group of hitters drawn at random *with replacement* from the real
    ones, and see how much k moves. Returns the 10th and 90th percentile of k
    per metric, and how often no signal was detectable at all."""
    rng = np.random.default_rng(seed)
    found = {metric: [] for metric in tables}
    for _ in range(resamples):
        for metric, events in tables.items():
            groups = [g for _, g in events.groupby("batter_id")]
            drawn = rng.integers(0, len(groups), size=len(groups))
            # a hitter drawn twice must count as two hitters, so give each draw a new id
            sample = pd.concat([groups[j].assign(batter_id=i) for i, j in enumerate(drawn)],
                               ignore_index=True)
            points = [p for n in sample_sizes(sample)
                      if (p := split_half(sample, n, splits, int(rng.integers(1 << 30)))) is not None]
            curve = pd.DataFrame(points, columns=["n", "hitters", "r_half", "r"])
            found[metric].append(fit_k(curve) if len(curve) >= 2 else K_MAX)
    rows = []
    for metric, ks in found.items():
        ks = np.array(ks)
        rows.append({"metric": metric,
                     "k_lo": float(np.percentile(ks, 10)),
                     "k_hi": float(np.percentile(ks, 90)),
                     "share_no_signal": float(np.mean(ks >= K_MAX * 0.99))})
    return pd.DataFrame(rows)


def shrink(numerator, denominator, league_value: float, k: float | None):
    """Pull a raw rate toward the league value by how little data is behind it:

        shrunk = (numerator + k * league) / (denominator + k)

    The same thing as adding k league-average events to every hitter. With no
    detectable signal (k is None) every hitter gets the league value."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if k is None:
        return np.full(numerator.shape, league_value)
    return (numerator + k * league_value) / (denominator + k)
