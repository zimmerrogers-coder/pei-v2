"""Stage 6 checks: split-half reliability and shrinkage, on simulated hitters
where the true answer is known."""

import numpy as np
import pandas as pd
import pytest

from app.pei import reliability


def simulated_events(hitters=300, events=120, mean=0.20, talent_sd=0.06, seed=5):
    """Each hitter has her own true rate; every event is a coin flip at that rate."""
    rng = np.random.default_rng(seed)
    talent = rng.normal(mean, talent_sd, hitters).clip(0.01, 0.99)
    return pd.DataFrame({
        "batter_id": np.repeat(np.arange(hitters), events),
        "game_id": "g",
        "value": (rng.random(hitters * events) < np.repeat(talent, events)).astype(float),
    })


def test_spearman_brown():
    assert reliability.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert reliability.spearman_brown(0.0) == 0.0


def test_recovers_a_known_k():
    # For a yes/no event, k = (average p(1-p) across hitters) / (variance of true talent).
    mean, sd = 0.20, 0.06
    true_k = (mean * (1 - mean) - sd ** 2) / sd ** 2          # about 43
    curve, summary = reliability.analyse({"k_rate": simulated_events(mean=mean, talent_sd=sd)},
                                         splits=40)
    row = summary.iloc[0]
    assert row["k"] == pytest.approx(true_k, rel=0.30)
    assert row["stabilization_n"] == pytest.approx(row["k"] * 0.7 / 0.3)
    # reliability rises with sample size
    assert curve["r"].iloc[-1] > curve["r"].iloc[0]


def test_pure_luck_has_no_usable_signal():
    # every hitter has the same true rate, so nothing separates them
    _, summary = reliability.analyse({"k_rate": simulated_events(talent_sd=0.0)}, splits=40)
    k = summary["k"].iloc[0]
    assert pd.isna(k) or k > 1000


def test_too_few_hitters_reports_nothing():
    events = simulated_events(hitters=reliability.MIN_HITTERS - 1)
    assert reliability.sample_sizes(events) == []
    assert reliability.split_half(events, 10) is None
    _, summary = reliability.analyse({"k_rate": events}, splits=10)
    assert pd.isna(summary["k"].iloc[0])


def test_split_uses_only_hitters_with_enough_events():
    few = simulated_events(hitters=20, events=6, seed=1)
    many = simulated_events(hitters=15, events=40, seed=2).assign(
        batter_id=lambda d: d["batter_id"] + 1000)
    point = reliability.split_half(pd.concat([few, many]), n=20, splits=5)
    assert point["hitters"] == 15 and point["n"] == 20


def test_shrink():
    # 4 strikeouts in 10 PA, league .130, k = 15  ->  (4 + 1.95) / 25
    assert reliability.shrink(4, 10, 0.130, 15) == pytest.approx(0.238)
    # no data at all gives the league value; a huge sample barely moves
    assert reliability.shrink(0, 0, 0.130, 15) == pytest.approx(0.130)
    assert reliability.shrink(400, 1000, 0.130, 15) == pytest.approx(0.396, abs=0.001)
    # no detectable signal: everyone is the league value
    assert list(reliability.shrink([4, 0], [10, 10], 0.130, None)) == [0.130, 0.130]


def test_bootstrap_range_brackets_the_estimate():
    events = simulated_events(hitters=120, events=60)
    _, summary = reliability.analyse({"k_rate": events}, splits=20)
    spread = reliability.bootstrap_k({"k_rate": events}, resamples=12, splits=10).iloc[0]
    assert spread["k_lo"] <= summary["k"].iloc[0] * 1.5
    assert spread["k_hi"] >= summary["k"].iloc[0] * 0.67
    assert spread["k_lo"] < spread["k_hi"]
