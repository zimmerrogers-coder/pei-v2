"""Reads one Dartfish video-tag export and lines it up with a TrackMan game.

A Dartfish row has no pitch id. What it does have is the TrackMan release speed
for that pitch, copied in by the tagging workflow, and the same row order as the
TrackMan file. So a tag file is matched to a game by checking that the two
speed sequences are identical, and then joined row by row.

TrackMan stays the source for what happened on the pitch. `find_conflicts`
only reports where the two disagree; it changes nothing.
"""

import csv
import re
from pathlib import Path

from app.importers.trackman import clean, parse_date, read_rows, to_num

_MARKERS = {"5. Game Pitch Result", "2. Hitters Name"}
SPEED_TOLERANCE = 0.01      # mph; the two files carry the same number

# Dartfish pitch result -> the TrackMan PitchCall values it agrees with
PITCH_RESULT = {
    "ball": {"BallCalled", "BallinDirt", "BallIntentional"},
    "strike called": {"StrikeCalled"},
    "strike swinging": {"StrikeSwinging"},
    "foul": {"FoulBallNotFieldable", "FoulBallFieldable", "FoulBall"},
    "ball in play": {"InPlay"},
    "hit by pitch": {"HitByPitch"},
}

# Dartfish hit result -> the TrackMan PlayResult values it agrees with.
# Anything not listed (Ground Out, Fly Out, Double Play, ...) is an out; a
# sacrifice is an out too, so "Ground Out" against "Sacrifice" is not a conflict.
HIT_RESULT = {
    "single": {"Single"},
    "double": {"Double"},
    "triple": {"Triple"},
    "home run": {"HomeRun"},
    "fielders choice": {"FieldersChoice"},
}
_OUT = {"Out", "Sacrifice"}

_JERSEY = re.compile(r"^\s*\d+\s+")


def is_dartfish(path: str) -> bool:
    if Path(path).suffix.lower() not in (".csv", ".txt"):
        return False
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        header = {h.strip() for h in next(csv.reader(fh), [])}
    return _MARKERS <= header


def strip_jersey(name) -> str | None:
    """'22 Avery Smith' -> 'Avery Smith'."""
    text = clean(name)
    return _JERSEY.sub("", text) if text else None


def _leading_int(value) -> int | None:
    match = re.match(r"\s*(\d+)", clean(value) or "")
    return int(match.group(1)) if match else None


def parse_dartfish(path: str) -> list[dict]:
    """One dict per tagged pitch, in file order."""
    tags = []
    for r in read_rows(path):
        if clean(r.get("5. Game Pitch Result")) is None:
            continue                      # not a pitch row
        tags.append({
            "game_date": parse_date(r.get("1. Date")),
            "hitter_name": strip_jersey(r.get("2. Hitters Name")),
            "hitter_team": clean(r.get("Team 2. Hitters Name")),
            "pitcher_team": clean(r.get("Team 1. Pitchers Name")),
            "rel_speed": to_num(r.get("RelSpeed")),
            "baserunning": clean(r.get("1. Baserunning Situation")),
            "pitch_result": clean(r.get("5. Game Pitch Result")),
            "hit_result": clean(r.get("8. Hit Results")),
            "hit_type": clean(r.get("8. Hit Type")),
            "hit_location": clean(r.get("8. Hit Location")),
            "hit_xy": clean(r.get("XY Hit Location")),
            "hitting_position": clean(r.get("Hitting Position")),
            "well_hit": clean(r.get("8. Well Hit")) is not None,
            "rbi": _leading_int(r.get("8. RBI")),
            "raw": {k: v for k, v in r.items() if k and clean(v) is not None},
        })
    if not tags:
        raise ValueError("That file has no tagged pitches.")
    return tags


def lines_up(tags: list[dict], pitches: list[dict]) -> bool:
    """True when the tag rows and a game's pitches are the same pitches in the
    same order: equal count and equal release speed on every row that has one."""
    if len(tags) != len(pitches):
        return False
    compared = 0
    for tag, pitch in zip(tags, pitches):
        a, b = tag["rel_speed"], pitch["rel_speed"]
        if a is None or b is None:
            continue
        if abs(a - b) > SPEED_TOLERANCE:
            return False
        compared += 1
    return compared >= len(tags) // 2     # most rows must actually have been checked


def find_conflicts(tags: list[dict], pitches: list[dict]) -> list[dict]:
    """Rows where the video tag and TrackMan disagree. `pitches` need pitch_no,
    pitch_call and play_result. TrackMan wins; this is a report for a human."""
    conflicts = []
    for tag, pitch in zip(tags, pitches):
        result = (tag["pitch_result"] or "").lower()
        if result in PITCH_RESULT and pitch["pitch_call"] not in PITCH_RESULT[result]:
            conflicts.append({"pitch_no": pitch["pitch_no"], "hitter": tag["hitter_name"],
                              "field": "pitch result", "dartfish": tag["pitch_result"],
                              "trackman": pitch["pitch_call"]})
        if tag["hit_result"] and pitch["pitch_call"] == "InPlay":
            agrees = HIT_RESULT.get(tag["hit_result"].lower(), _OUT)
            if pitch["play_result"] not in agrees:
                conflicts.append({"pitch_no": pitch["pitch_no"], "hitter": tag["hitter_name"],
                                  "field": "hit result", "dartfish": tag["hit_result"],
                                  "trackman": pitch["play_result"]})
    return conflicts
