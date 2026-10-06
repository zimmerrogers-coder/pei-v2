"""Stage 5 checks: the expected-stats model and its comparison with v1's buckets."""

import warnings

import numpy as np
import pandas as pd
import psycopg
import pytest

from app import db
from app.pei import config, plate_appearances, xstats

SETTINGS = config.load()
warnings.filterwarnings("ignore")


def simulated_balls(n=600, seed=3, games=6):
    """Batted balls where the truth is known: harder and higher means better."""
    rng = np.random.default_rng(seed)
    ev = rng.normal(60, 12, n).clip(20, 95)
    la = rng.normal(10, 25, n).clip(-70, 70)
    quality = (ev - 60) / 12 + 0.6 * (1 - ((la - 22) / 25) ** 2)
    roll = quality + rng.normal(0, 1.0, n)
    outcome = np.select([roll > 2.6, roll > 1.9, roll > 0.9], ["HR", "2B", "1B"], "OUT")
    balls = pd.DataFrame({
        "game_id": [f"g{i % games}" for i in range(n)], "in_play": True,
        "exit_speed": ev, "launch_angle": la, "outcome": outcome,
    })
    balls["woba_value"] = balls["outcome"].map(SETTINGS.woba_weights).fillna(0.0)
    return balls


def test_probabilities_are_valid_and_follow_contact_quality():
    balls = simulated_balls()
    model = xstats.SmoothModel().fit(balls, SETTINGS)
    proba = model.proba(balls)
    assert proba.shape == (len(balls), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)

    weak, crushed = pd.DataFrame({"exit_speed": [40.0, 80.0], "launch_angle": [-10.0, 25.0]}).pipe(
        model.xwobacon)
    assert crushed > weak + 0.5


def test_scoring_keeps_the_class_order():
    """A method that knows every answer must score a log loss near zero. This
    fails if the probability columns and the labels are ever lined up wrongly."""
    balls = simulated_balls(300)

    class Oracle(xstats.NoInformation):
        name = "no information (league average)"      # takes that method's slot in the table

        def proba(self, test):
            exact = np.full((len(test), 4), 1e-9)
            for row, label in enumerate(xstats.contact_class(test["outcome"])):
                exact[row, xstats.CLASSES.index(label)] = 1.0
            return exact

    original = xstats.NoInformation
    xstats.NoInformation = Oracle
    try:
        scores = xstats.evaluate(balls, SETTINGS, repeats=1).set_index("method")
    finally:
        xstats.NoInformation = original
    assert scores.loc["no information (league average)", "log_loss"] < 0.001


def test_smooth_model_beats_buckets_on_simulated_balls():
    scores = xstats.evaluate(simulated_balls(), SETTINGS, repeats=1).set_index("method")
    smooth = scores.loc["smooth EV/LA model", "log_loss"]
    assert smooth < scores.loc["v1 buckets, values fitted", "log_loss"]
    assert smooth < scores.loc["no information (league average)", "log_loss"]


def test_out_of_fold_never_uses_the_balls_own_game():
    balls = simulated_balls()
    values = xstats.out_of_fold(balls, SETTINGS)
    assert values.notna().all() and values.between(0, 2.1).all()

    # change every result in one game: that game's own expected values must not move
    changed = balls.copy()
    in_game = changed["game_id"] == "g0"
    changed.loc[in_game, "outcome"] = "HR"
    changed.loc[in_game, "woba_value"] = SETTINGS.woba_weights["HR"]
    after = xstats.out_of_fold(changed, SETTINGS)
    assert np.allclose(values[in_game], after[in_game])
    assert not np.allclose(values[~in_game], after[~in_game])


def test_too_little_data_gives_no_expected_stats():
    values = xstats.out_of_fold(simulated_balls(40), SETTINGS)
    assert values.isna().all()


def test_real_balls_smooth_model_beats_v1_buckets():
    try:
        conn = db.connect()
    except psycopg.OperationalError as err:
        pytest.skip(f"database not reachable ({err})")
    if conn.execute("SELECT count(*) FROM core.pitch").fetchone()[0] == 0:
        pytest.skip("no game files imported")
    pa, _ = plate_appearances.derive(plate_appearances.load_pitches(conn), SETTINGS)
    conn.close()

    scores = xstats.evaluate(pa, SETTINGS, repeats=1).set_index("method")
    assert scores["balls"].iloc[0] > 0
    assert (scores.loc["smooth EV/LA model", "log_loss"]
            < scores.loc["v1 buckets, values fitted", "log_loss"]
            < scores.loc["no information (league average)", "log_loss"])
    assert (scores.loc["smooth EV/LA model", "rmse"]
            < scores.loc["v1 buckets, hand-set values", "rmse"])
