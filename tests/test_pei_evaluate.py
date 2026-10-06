"""Stage 8 checks: the holdout evaluation's splits, baselines and scoring."""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

from app.pei import config, evaluate, model

SETTINGS = config.load()
warnings.filterwarnings("ignore")

DATES = {f"g{i:02d}": dt.date(2026, 3, 1) + dt.timedelta(days=i) for i in range(12)}


def test_chronological_splits_train_on_the_earliest_games():
    splits = evaluate.chronological_splits(DATES)
    assert [len(s["train"]) for s in splits] == [3, 6, 9]
    assert splits[0]["train"] == ["g00", "g01", "g02"]
    # a later origin only ever adds games
    assert set(splits[0]["train"]) < set(splits[1]["train"]) < set(splits[2]["train"])


def test_random_halves_are_repeatable_and_differ():
    first = evaluate.random_half_splits(DATES, count=4, seed=0)
    again = evaluate.random_half_splits(DATES, count=4, seed=0)
    assert [s["train"] for s in first] == [s["train"] for s in again]
    assert all(len(s["train"]) == 6 for s in first)
    assert len({tuple(s["train"]) for s in first}) > 1
    # they come in mirrored pairs: the second trains on exactly what the first held out
    for a, b in zip(first[::2], first[1::2]):
        assert set(a["train"]) | set(b["train"]) == set(DATES)
        assert not set(a["train"]) & set(b["train"])


def test_luck_floor():
    # one hitter, 25 held-out PA, each PA varying by 0.5: even perfect knowledge misses by 0.1
    rows = pd.DataFrame({"test_pa": [25]})
    assert evaluate.luck_floor(rows, per_pa_sd=0.5) == pytest.approx(0.1)


def test_baselines():
    totals = pd.DataFrame({"pa": [10, 40], "woba_num": [6.0, 12.0], "xwoba_num": [4.0, 16.0]},
                          index=pd.Index([1, 2], name="batter_id"))
    k = {"woba": 20.0, "xwoba": 10.0}
    league = {"woba": 0.350, "xwoba": 0.360}
    out = evaluate.baseline_predictions(totals, k, league, [1, 2, 99])

    assert out.loc[1, "raw wOBA to date"] == pytest.approx(0.600)
    assert out.loc[1, "shrunk wOBA"] == pytest.approx((6 + 20 * 0.35) / 30)
    assert out.loc[2, "shrunk xwOBA"] == pytest.approx((16 + 10 * 0.36) / 50)
    # hitter 99 was never seen: every method falls back to the league
    assert out.loc[99, "raw wOBA to date"] == 0.350
    assert out.loc[99, "shrunk wOBA"] == pytest.approx(0.350)
    assert out.loc[99, "shrunk xwOBA"] == pytest.approx(0.360)
    assert (out["league average"] == 0.350).all()


def test_baseline_with_no_detectable_signal_is_the_league():
    totals = pd.DataFrame({"pa": [10], "woba_num": [6.0], "xwoba_num": [4.0]}, index=[1])
    out = evaluate.baseline_predictions(totals, {"woba": None, "xwoba": None},
                                        {"woba": 0.35, "xwoba": 0.36}, [1])
    assert out.loc[1, "shrunk wOBA"] == 0.35


def test_feature_without_signal_is_dropped():
    totals = pd.DataFrame({"whiffs": [2, 10], "swings": [10, 20], "chases": [1, 5],
                           "chase_pitches": [10, 20]}, index=[1, 2])
    k = {"whiff_rate": 30.0, "chase_rate": None}
    league = {"whiff_rate": 0.2, "chase_rate": 0.3}
    out = evaluate.shrunk_features(totals, k, league, {"k": ["whiff_rate"], "bb": ["chase_rate"]})
    assert list(out) == ["k"]
    assert out["k"].loc[1, "whiff_rate"] == pytest.approx((2 + 30 * 0.2) / 40)


def test_score_weights_by_plate_appearances():
    rows = pd.DataFrame({
        "test_pa": [30, 2], "actual": [0.300, 1.000],
        "good on the regular": [0.300, 0.300],      # exact on 30 PA, far off on 2 PA
        "good on the sub": [0.500, 1.000],          # exact on 2 PA, off on 30 PA
        "good on the regular lo": [0.2, 0.2], "good on the regular hi": [0.4, 0.4],
    })
    scores = evaluate.score(rows, ["good on the regular", "good on the sub"]).set_index("method")
    assert scores.loc["good on the regular", "rmse"] < scores.loc["good on the sub", "rmse"]
    assert scores.loc["good on the regular", "rmse"] == pytest.approx(np.sqrt(2 * 0.49 / 32))
    assert scores.loc["good on the regular", "coverage_80"] == 0.5      # 1 of 2 inside
    assert pd.isna(scores.loc["good on the sub", "coverage_80"])
    assert scores.loc["good on the sub", "bias"] > 0                    # predicted too high


def test_compare_reports_a_clear_winner_and_a_coin_flip():
    rng = np.random.default_rng(0)
    actual = rng.normal(0.35, 0.08, 60)
    rows = pd.DataFrame({"test_pa": 20, "actual": actual,
                         "close": actual + rng.normal(0, 0.02, 60),
                         "also close": actual + rng.normal(0, 0.02, 60),
                         "far": actual + rng.normal(0, 0.15, 60)})
    clear = evaluate.compare(rows, "close", "far", resamples=500)
    assert clear["rmse_gain"] > 0 and clear["share_better"] > 0.99 and clear["gain_lo"] > 0
    toss_up = evaluate.compare(rows, "close", "also close", resamples=500)
    assert toss_up["gain_lo"] < 0 < toss_up["gain_hi"]


def test_model_predictions_on_simulated_hitters():
    pa, _ = model.simulate(n_hitters=40, seed=4, settings=SETTINGS)
    test_pa = pd.Series({0: 4, 1: 40, 999: 20})            # 999 was never seen
    predicted, diagnostics = evaluate.model_predictions(pa, SETTINGS, test_pa, draws=400,
                                                        tune=600, n_sim=600)
    assert diagnostics["divergences"] == 0
    assert list(predicted.index) == [0, 1, 999]
    assert (predicted["lo"] <= predicted["pred"]).all() and (predicted["pred"] <= predicted["hi"]).all()
    assert predicted["pred"].between(0.15, 0.70).all()
    width = predicted["hi"] - predicted["lo"]
    # the same kind of hitter over more plate appearances has less luck in her result
    assert width[1] < width[0]
    # an unseen hitter is predicted near the league
    assert abs(predicted.loc[999, "pred"] - predicted["pred"].mean()) < 0.08
