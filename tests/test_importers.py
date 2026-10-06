"""Stage 2 checks: the two file readers, the pitch-for-pitch join, and the database import.

The tests build their own small files, so no real game data is needed.
"""

import csv

import psycopg
import pytest

from app import db
from app.importers import dartfish, store, trackman


# pitch_no, batter, batter team, call, KorBB, play result, speed, exit speed, angle
PITCHES = [
    (1, "Smith, Avery ", "HOME_SB", "BallCalled", "Undefined", "Undefined", "70.11", "", ""),
    (2, "Smith, Avery", "HOME_SB", "InPlay", "Undefined", "Single", "69.50", "71.2", "14.0"),
    (3, "Jones, Riley", "AWAY_SB", "StrikeSwinging", "Strikeout", "Undefined", "64.02", "", ""),
    (4, "Brown, Casey", "AWAY_SB", "InPlay", "Undefined", "Error", "63.40", "55.0", "-3.5"),
]


def write_trackman(path, uid_prefix="uid", game_id="20260313-Test-1"):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["PitchNo", "Date", "Pitcher", "PitcherTeam", "Batter", "BatterTeam",
                    "BatterSide", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes",
                    "PitchCall", "KorBB", "PlayResult", "RelSpeed", "ExitSpeed", "Angle",
                    "HomeTeam", "AwayTeam", "Stadium", "GameID", "PitchUID", "System"])
        for no, batter, team, call, korbb, result, speed, ev, la in PITCHES:
            pitcher_team = "AWAY_SB" if team == "HOME_SB" else "HOME_SB"
            pitcher = "Davis, Jordan" if pitcher_team == "AWAY_SB" else "Miller, Morgan"
            w.writerow([no, "3/13/2026", pitcher, pitcher_team, batter, team, "Left",
                        1, "Top", 0, 0, 0, call, korbb, result, speed, ev, la,
                        "AWAY_SB", "HOME_SB", "Test", game_id, f"{uid_prefix}-{no}", "v3"])
    return str(path)


def write_dartfish(path, speeds=("70.11", "69.50", "64.02", "63.40")):
    results = ["Ball", "Ball in Play", "Strike Swinging", "Ball in Play"]
    hits = ["", "Single", "", "Single"]            # last row disagrees with TrackMan's Error
    names = ["22 Avery Smith", "22 Avery Smith", "21 Riley Jones", "43 Casey Brown"]
    teams = ["Home U", "Home U", "Away U", "Away U"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Name", "1. Baserunning Situation", "1. Date", "2. Hitters Name",
                    "5. Game Pitch Result", "RelSpeed", "Team 1. Pitchers Name",
                    "Team 2. Hitters Name", "8. Hit Results", "Hitting Position", "8. RBI"])
        for i in range(4):
            other = "Away U" if teams[i] == "Home U" else "Home U"
            w.writerow([f"PITCH ({i + 1})", "R1", "3/13/2026", names[i], results[i], speeds[i],
                        other, teams[i], hits[i], "Slap" if i == 1 else "", "1 RBI" if i == 1 else ""])
    return str(path)


# ---------------------------------------------------------------- parsing

def test_trackman_parse(tmp_path):
    path = write_trackman(tmp_path / "game.csv")
    assert trackman.is_trackman(path)
    parsed = trackman.parse_trackman(path)

    assert parsed.game["game_id"] == "20260313-Test-1"
    assert parsed.game["game_date"].isoformat() == "2026-03-13"
    assert parsed.game["season"] == 2026
    assert parsed.teams == {"HOME_SB", "AWAY_SB"}
    # "Smith, Avery " and "Smith, Avery" are one player, stored "First Last"
    assert ("Avery Smith", "HOME_SB") in parsed.players
    assert len([k for k in parsed.players if k[0] == "Avery Smith"]) == 1

    ball, single = parsed.pitches[0], parsed.pitches[1]
    assert ball["kor_bb"] is None and ball["play_result"] is None    # "Undefined" -> empty
    assert ball["exit_speed"] is None
    assert single["exit_speed"] == 71.2 and single["launch_angle"] == 14.0
    assert single["raw"]["PitchUID"] == "uid-2"


def test_dartfish_parse_and_join(tmp_path):
    pitches = trackman.parse_trackman(write_trackman(tmp_path / "game.csv")).pitches
    path = write_dartfish(tmp_path / "tags.csv")
    assert dartfish.is_dartfish(path) and not trackman.is_trackman(path)
    tags = dartfish.parse_dartfish(path)

    assert tags[0]["hitter_name"] == "Avery Smith"
    assert tags[1]["hitting_position"] == "Slap" and tags[1]["rbi"] == 1
    assert dartfish.lines_up(tags, pitches)

    conflicts = dartfish.find_conflicts(tags, pitches)
    assert [(c["pitch_no"], c["dartfish"], c["trackman"]) for c in conflicts] == [(4, "Single", "Error")]


def test_dartfish_from_another_game_does_not_line_up(tmp_path):
    pitches = trackman.parse_trackman(write_trackman(tmp_path / "game.csv")).pitches
    wrong_speed = dartfish.parse_dartfish(
        write_dartfish(tmp_path / "a.csv", speeds=("70.11", "66.00", "64.02", "63.40")))
    assert not dartfish.lines_up(wrong_speed, pitches)
    assert not dartfish.lines_up(wrong_speed[:3], pitches)           # different pitch count


# ---------------------------------------------------------------- database

@pytest.fixture
def conn():
    try:
        connection = db.connect(db.TEST_DB_NAME)
    except psycopg.OperationalError as err:
        pytest.skip(f"test database not reachable; run scripts/create_database.py ({err})")
    db.apply_schema(connection)
    connection.execute("TRUNCATE core.import_file, core.game, core.player, core.team CASCADE")
    connection.commit()
    yield connection
    connection.close()


def _count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_import_roundtrip(conn, tmp_path):
    game = write_trackman(tmp_path / "game.csv")
    summary = store.import_file(conn, game)
    assert summary["source"] == "trackman" and summary["pitches"] == 4
    assert _count(conn, "core.pitch") == 4 and _count(conn, "core.player") == 5   # 3 hitters, 2 pitchers

    tags = store.import_file(conn, write_dartfish(tmp_path / "tags.csv"))
    assert tags["game_id"] == "20260313-Test-1" and len(tags["conflicts"]) == 1
    assert _count(conn, "core.dartfish_pitch") == 4
    names = dict(conn.execute("SELECT team_code, display_name FROM core.team").fetchall())
    assert names == {"HOME_SB": "Home U", "AWAY_SB": "Away U"}

    # deleting the TrackMan import removes everything that came with it
    store.delete_import(conn, summary["import_id"])
    for table in ("core.pitch", "core.dartfish_pitch", "core.game", "core.import_file"):
        assert _count(conn, table) == 0


def test_duplicates_are_refused(conn, tmp_path):
    game = write_trackman(tmp_path / "game.csv")
    store.import_file(conn, game)

    copy = tmp_path / "game (1).csv"                    # same bytes, new name
    copy.write_bytes(Path(game).read_bytes())
    with pytest.raises(store.ImportRefused, match="already imported"):
        store.import_file(conn, str(copy))

    # a different file that contains the same pitches
    edited = write_trackman(tmp_path / "edited.csv", game_id="20260313-Test-1")
    Path(edited).write_text(Path(edited).read_text() + "\n")
    with pytest.raises(store.ImportRefused, match="already in the database"):
        store.import_file(conn, edited)
    assert _count(conn, "core.pitch") == 4 and _count(conn, "core.import_file") == 1


def test_dartfish_needs_its_trackman_game(conn, tmp_path):
    with pytest.raises(store.ImportRefused, match="Import the TrackMan file first"):
        store.import_file(conn, write_dartfish(tmp_path / "tags.csv"))
