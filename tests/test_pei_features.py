"""Stage 4 checks: plate appearances, the outcome mapping, and the feature counts.

The small tests use hand-built pitches where the right answer is obvious. The
real-data tests run only when PEI_DATA_DIR points at a folder of game files and
those games are imported; the last one recounts every hitter straight from the
raw CSV files, without using any of the app's code, and compares.
"""

import os
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from app import db
from app.pei import build, config, features, plate_appearances, validate

SETTINGS = config.load()
# Folder of real TrackMan/Dartfish game files; the real-data tests skip without it.
DATA_DIR = Path(os.environ.get("PEI_DATA_DIR", "game_files"))


def pitch(pa, batter, call, korbb=None, result=None, hit_type=None, ev=None, la=None,
          balls=0, strikes=0, height=2.0, side=0.0, tag=None):
    return dict(game_id="g1", inning=1, top_bottom="Top", pa_of_inning=pa, batter_id=batter,
                pitcher_id=99, balls=balls, strikes=strikes, pitch_call=call, kor_bb=korbb,
                play_result=result, tagged_hit_type=hit_type, exit_speed=ev, launch_angle=la,
                plate_loc_height=height, plate_loc_side=side, hitting_position=tag)


def frame(rows):
    out = pd.DataFrame(rows)
    out["pitch_no"] = range(1, len(out) + 1)
    out["pitch_uid"] = [f"u{i}" for i in out["pitch_no"]]
    return out


def outcomes(rows):
    pa, problems = plate_appearances.derive(frame(rows), SETTINGS)
    return pa, problems


def test_outcome_mapping():
    pa, problems = outcomes([
        pitch(1, 1, "InPlay", result="Single", hit_type="LineDrive", ev=70, la=15),
        pitch(2, 2, "InPlay", result="HomeRun", hit_type="FlyBall", ev=75, la=25),
        pitch(3, 3, "InPlay", result="Error", hit_type="GroundBall", ev=50, la=-5),
        pitch(4, 4, "InPlay", result="Sacrifice", hit_type="Bunt"),
        pitch(5, 5, "InPlay", result="Sacrifice", hit_type="FlyBall", ev=60, la=35),
        pitch(6, 6, "InPlay", result="Sacrifice", hit_type="GroundBall", tag="Bunt"),
        pitch(7, 7, "HitByPitch", korbb="Walk"),          # TrackMan marks some HBP as walks
        pitch(8, 8, "BallCalled", korbb="Walk", balls=3),
        pitch(9, 9, "StrikeSwinging", korbb="Strikeout", strikes=2),
        pitch(10, 10, "InPlay", result="FieldersChoice", hit_type="GroundBall", ev=40, la=0),
    ])
    assert problems.empty
    assert pa["outcome"].tolist() == ["1B", "HR", "ROE", "SH", "SF", "SH", "HBP", "BB", "K", "FC"]
    assert pa["woba_value"].round(3).tolist() == [0.88, 2.031, 0, 0, 0, 0, 0.72, 0.69, 0, 0]
    assert pa["is_ab"].tolist() == [True, True, True, False, False, False, False, False, True, True]
    assert pa["is_hit"].sum() == 2          # reaching on an error is not a hit


def test_contact_flags_use_the_agreed_thresholds():
    pa, _ = outcomes([
        pitch(1, 1, "InPlay", result="Out", ev=62.9, la=8),     # just under hard-hit; sweet spot edge
        pitch(2, 1, "InPlay", result="Out", ev=63.0, la=32),    # hard hit; sweet spot edge
        pitch(3, 1, "InPlay", result="Out", ev=70.0, la=20),    # barrel
        pitch(4, 1, "InPlay", result="Out", ev=80.0, la=33),    # hard, but too steep for either
        pitch(5, 1, "InPlay", result="Out", hit_type="Bunt"),   # not measured
    ])
    assert pa["hard_hit"].tolist()[:4] == [False, True, True, True]
    assert pa["sweet_spot"].tolist()[:4] == [True, True, True, False]
    assert pa["barrel"].tolist()[:4] == [False, False, True, False]
    assert pa.loc[4, ["hard_hit", "sweet_spot", "barrel"]].isna().all()   # unknown, not False


def test_unfinished_plate_appearance_is_left_out_and_reported():
    pa, problems = outcomes([
        pitch(1, 1, "BallCalled"),
        pitch(1, 1, "InPlay", result="Out", ev=60, la=10, balls=1),
        pitch(2, 2, "BallCalled"),                      # runner caught stealing ends the inning
    ])
    assert len(pa) == 1 and pa["n_pitches"].iloc[0] == 2
    assert len(problems) == 1 and "no ending pitch" in problems["reason"].iloc[0]


def test_pinch_hitter_mid_count():
    # replaced with two strikes, substitute strikes out -> charged to the first batter
    pa, _ = outcomes([
        pitch(1, 1, "StrikeCalled"), pitch(1, 1, "FoulBall", strikes=1),
        pitch(1, 2, "StrikeSwinging", korbb="Strikeout", strikes=2),
    ])
    assert pa["batter_id"].tolist() == [1] and pa["outcome"].tolist() == ["K"]
    # replaced with one strike, substitute singles -> the substitute's
    pa, _ = outcomes([
        pitch(1, 1, "StrikeCalled"),
        pitch(1, 2, "InPlay", result="Single", ev=65, la=12, strikes=1),
    ])
    assert pa["batter_id"].tolist() == [2]


def test_discipline_counts_and_rates():
    rows = [
        pitch(1, 1, "StrikeSwinging", height=2.0, side=0.0),              # zone swing, whiff
        pitch(1, 1, "BallCalled", height=3.5, side=0.0, strikes=1),       # outside, taken
        pitch(1, 1, "FoulBallNotFieldable", height=0.5, side=0.0, balls=1, strikes=1),   # chase
        pitch(1, 1, "StrikeCalled", height=2.0, side=0.5, balls=1, strikes=2),           # zone, taken
        pitch(2, 1, "BallCalled", height=None, side=None),                # no location
        pitch(2, 1, "InPlay", result="Double", ev=72, la=22, height=2.0, side=-0.87, balls=1),
    ]
    rows[3]["kor_bb"] = "Strikeout"
    pitches = frame(rows)
    pa, _ = plate_appearances.derive(pitches, SETTINGS)
    games = features.hitter_game(pa, pitches, SETTINGS)
    row = features.rates(games).iloc[0]

    assert (row.pa, row.k, row.b2, row.bip, row.bip_measured) == (2, 1, 1, 1, 1)
    assert (row.pitches, row.pitches_located, row.swings, row.whiffs) == (6, 5, 3, 1)
    assert (row.zone_pitches, row.zone_swings, row.chase_pitches, row.chases) == (3, 2, 2, 1)
    assert row.woba == pytest.approx(1.247 / 2)          # per plate appearance
    assert row.k_rate == 0.5 and row.chase_rate == 0.5
    assert row.whiff_rate == pytest.approx(1 / 3)
    assert row.barrel_rate == 1.0
    assert all(check.empty for check in validate.check_all(pa, games).values())


def test_rate_with_no_denominator_is_empty_not_zero():
    counts = pd.DataFrame([{c: 0 for c in features.COUNT_COLUMNS}])
    assert features.rates(counts)[list(features.RATES)].isna().all(axis=None)


# ---------------------------------------------------------------- real data

@pytest.fixture
def real_conn():
    try:
        connection = db.connect()
    except psycopg.OperationalError as err:
        pytest.skip(f"database not reachable ({err})")
    if connection.execute("SELECT count(*) FROM core.pitch").fetchone()[0] == 0:
        pytest.skip("no game files imported")
    yield connection
    connection.close()


def test_real_build_passes_every_check(real_conn):
    result = build.build(real_conn)
    assert result["plate_appearances"] > 0
    failed = [name for name, rows in result["checks"].items() if not rows.empty]
    assert failed == []


@pytest.mark.skipif(not DATA_DIR.exists(), reason="real game files not on this machine")
def test_three_hitters_recounted_from_the_raw_files(real_conn):
    """Count by hand from the CSVs, the way v1 did, and compare with the tables."""
    raw = pd.concat(
        [pd.read_csv(p, dtype=str, encoding="utf-8-sig") for p in DATA_DIR.glob("*.csv")
         if "PitchUID" in pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns],
        ignore_index=True)
    raw["Batter"] = raw["Batter"].str.strip()

    build.build(real_conn)
    totals = build.load_hitter_games(real_conn)
    totals = features.rates(
        totals.groupby("full_name", as_index=False)[features.COUNT_COLUMNS].sum()
    ).set_index("full_name")

    for csv_name in sorted(raw["Batter"].dropna().unique()):
        last, _, first = csv_name.partition(",")
        name = f"{first.strip()} {last.strip()}" if first else csv_name
        if name not in totals.index:
            continue
        mine = raw[raw["Batter"] == csv_name]
        result = mine["PlayResult"]
        in_play = mine["PitchCall"] == "InPlay"
        ev = pd.to_numeric(mine.loc[in_play, "ExitSpeed"], errors="coerce").dropna()
        expected = {
            "b1": (result == "Single").sum(), "b2": (result == "Double").sum(),
            "b3": (result == "Triple").sum(), "hr": (result == "HomeRun").sum(),
            "bb": ((mine["KorBB"] == "Walk") & (mine["PitchCall"] != "HitByPitch")).sum(),
            "hbp": (mine["PitchCall"] == "HitByPitch").sum(),
            "k": (mine["KorBB"] == "Strikeout").sum(),
            "bip": in_play.sum(), "bip_measured": len(ev),
            "hard_hit": (ev >= 63).sum(),
            "whiffs": (mine["PitchCall"] == "StrikeSwinging").sum(),
        }
        got = totals.loc[name]
        assert {k: int(got[k]) for k in expected} == {k: int(v) for k, v in expected.items()}, name
        pa = expected["bip"] + expected["bb"] + expected["hbp"] + expected["k"]
        woba = (0.69 * expected["bb"] + 0.72 * expected["hbp"] + 0.88 * expected["b1"]
                + 1.247 * expected["b2"] + 1.578 * expected["b3"] + 2.031 * expected["hr"]) / pa
        assert int(got["pa"]) == pa, name
        assert got["woba"] == pytest.approx(woba, abs=1e-6), name
