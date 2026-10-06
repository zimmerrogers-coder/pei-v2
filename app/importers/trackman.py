"""Reads one TrackMan game file into plain Python values. No database here:
parsing and storing are kept apart so the parsing can be tested on its own.
"""

import csv
import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

# TrackMan column -> (our column, converter). Everything else stays in `raw`.
_INT, _NUM, _TEXT = "int", "num", "text"
COLUMNS = {
    "PitchNo": ("pitch_no", _INT),
    "Inning": ("inning", _INT),
    "Top/Bottom": ("top_bottom", _TEXT),
    "PAofInning": ("pa_of_inning", _INT),
    "PitchofPA": ("pitch_of_pa", _INT),
    "Outs": ("outs", _INT),
    "Balls": ("balls", _INT),
    "Strikes": ("strikes", _INT),
    "TaggedPitchType": ("tagged_pitch_type", _TEXT),
    "PitchCall": ("pitch_call", _TEXT),
    "KorBB": ("kor_bb", _TEXT),
    "PlayResult": ("play_result", _TEXT),
    "TaggedHitType": ("tagged_hit_type", _TEXT),
    "OutsOnPlay": ("outs_on_play", _INT),
    "RunsScored": ("runs_scored", _INT),
    "RelSpeed": ("rel_speed", _NUM),
    "SpinRate": ("spin_rate", _NUM),
    "PlateLocHeight": ("plate_loc_height", _NUM),
    "PlateLocSide": ("plate_loc_side", _NUM),
    "ExitSpeed": ("exit_speed", _NUM),
    "Angle": ("launch_angle", _NUM),
    "Direction": ("direction", _NUM),
    "Distance": ("distance", _NUM),
}

# TrackMan writes these where it has nothing to say.
_EMPTY = {"", "undefined", "nan", "null"}


@dataclass
class ParsedGame:
    game: dict                                    # game_id, game_date, season, home_team, away_team, stadium
    teams: set = field(default_factory=set)       # team codes
    players: dict = field(default_factory=dict)   # (full_name, team_code) -> bats ('L'/'R'/None)
    pitches: list = field(default_factory=list)   # one dict per pitch


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        return list(csv.DictReader(fh))


def is_trackman(path: str) -> bool:
    """TrackMan files carry PitchUID and GameID; Yakkertech files do not."""
    if Path(path).suffix.lower() != ".csv":
        return False
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        header = set(next(csv.reader(fh), []))
    return {"PitchUID", "GameID", "PitchCall"} <= header


def clean(value) -> str | None:
    text = " ".join(str(value or "").split())
    return None if text.lower() in _EMPTY else text


def to_num(value) -> float | None:
    text = clean(value)
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def to_int(value) -> int | None:
    number = to_num(value)
    return int(number) if number is not None else None


def full_name(name) -> str | None:
    """'Smith, Avery ' -> 'Avery Smith'. Names without a comma are kept."""
    text = clean(name)
    if text is None:
        return None
    if "," in text:
        last, first = (part.strip() for part in text.split(",", 1))
        text = f"{first} {last}".strip()
    return text


def side(value) -> str | None:
    text = (clean(value) or "").upper()
    return text[0] if text[:1] in ("L", "R") else None


def parse_date(value) -> date | None:
    text = clean(value)
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(text or "", fmt).date()
        except ValueError:
            continue
    return None


_CONVERT = {_INT: to_int, _NUM: to_num, _TEXT: clean}


def parse_trackman(path: str) -> ParsedGame:
    rows = read_rows(path)
    if not rows:
        raise ValueError("That file has no pitches.")

    game_ids = {clean(r.get("GameID")) for r in rows}
    if len(game_ids) != 1 or None in game_ids:
        raise ValueError(f"Expected one GameID in the file, found {sorted(map(str, game_ids))}.")

    first = rows[0]
    game_date = parse_date(first.get("Date"))
    if game_date is None:
        raise ValueError(f"Could not read the game date {first.get('Date')!r}.")

    parsed = ParsedGame(game={
        "game_id": game_ids.pop(),
        "game_date": game_date,
        "season": game_date.year,
        "home_team": clean(first.get("HomeTeam")),
        "away_team": clean(first.get("AwayTeam")),
        "stadium": clean(first.get("Stadium")),
    })
    parsed.teams.update(t for t in (parsed.game["home_team"], parsed.game["away_team"]) if t)

    for r in rows:
        pitch_uid = clean(r.get("PitchUID"))
        if pitch_uid is None:
            raise ValueError("A pitch has no PitchUID.")

        batter = (full_name(r.get("Batter")), clean(r.get("BatterTeam")))
        pitcher = (full_name(r.get("Pitcher")), clean(r.get("PitcherTeam")))
        bats = side(r.get("BatterSide"))
        for key, hand in ((batter, bats), (pitcher, None)):
            if key[0]:
                parsed.players[key] = parsed.players.get(key) or hand
                if key[1]:
                    parsed.teams.add(key[1])

        pitch = {ours: _CONVERT[kind](r.get(theirs)) for theirs, (ours, kind) in COLUMNS.items()}
        pitch.update(
            pitch_uid=pitch_uid,
            batter=batter if batter[0] else None,
            pitcher=pitcher if pitcher[0] else None,
            batter_side=bats,
            raw={k: v for k, v in r.items() if k and clean(v) is not None},
        )
        if pitch["pitch_no"] is None:
            raise ValueError(f"Pitch {pitch_uid} has no PitchNo.")
        parsed.pitches.append(pitch)

    parsed.pitches.sort(key=lambda p: p["pitch_no"])
    return parsed
