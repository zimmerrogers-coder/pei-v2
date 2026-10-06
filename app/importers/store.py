"""Writes parsed files into the core tables. Each import is one transaction:
either the whole file goes in or none of it does.
"""

from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from app.importers import dartfish, trackman

_PITCH_COLUMNS = [ours for ours, _ in trackman.COLUMNS.values()]


class ImportRefused(ValueError):
    """The file was understood but must not be imported (duplicate, no match)."""


def _register_file(conn, source: str, path: str, row_count: int) -> int:
    digest = trackman.file_hash(path)
    seen = conn.execute(
        "SELECT filename FROM core.import_file WHERE source = %s AND file_hash = %s",
        (source, digest)).fetchone()
    if seen:
        raise ImportRefused(f"This file was already imported as '{seen[0]}'.")
    return conn.execute(
        "INSERT INTO core.import_file (source, filename, file_hash, row_count) "
        "VALUES (%s, %s, %s, %s) RETURNING import_id",
        (source, Path(path).name, digest, row_count)).fetchone()[0]


def _player_ids(conn, players: dict) -> dict:
    """(full_name, team_code) -> player_id, creating players that are new."""
    ids = {}
    for (name, team), bats in players.items():
        row = conn.execute(
            "SELECT player_id FROM core.player WHERE full_name = %s "
            "AND team_code IS NOT DISTINCT FROM %s", (name, team)).fetchone()
        if row is None:
            row = conn.execute(
                "INSERT INTO core.player (full_name, team_code, bats) VALUES (%s, %s, %s) "
                "RETURNING player_id", (name, team, bats)).fetchone()
        elif bats:
            conn.execute("UPDATE core.player SET bats = COALESCE(bats, %s) WHERE player_id = %s",
                         (bats, row[0]))
        ids[(name, team)] = row[0]
    return ids


def import_trackman(conn: psycopg.Connection, path: str) -> dict:
    parsed = trackman.parse_trackman(path)
    try:
        import_id = _register_file(conn, "trackman", path, len(parsed.pitches))

        uids = [p["pitch_uid"] for p in parsed.pitches]
        already = conn.execute(
            "SELECT count(*) FROM core.pitch WHERE pitch_uid = ANY(%s)", (uids,)).fetchone()[0]
        if already:
            raise ImportRefused(
                f"{already} of this file's {len(uids)} pitches are already in the database "
                f"(game {parsed.game['game_id']}).")

        for code in sorted(parsed.teams):
            conn.execute("INSERT INTO core.team (team_code) VALUES (%s) ON CONFLICT DO NOTHING",
                         (code,))
        conn.execute(
            "INSERT INTO core.game (game_id, game_date, season, home_team, away_team, stadium) "
            "VALUES (%(game_id)s, %(game_date)s, %(season)s, %(home_team)s, %(away_team)s, "
            "%(stadium)s) ON CONFLICT (game_id) DO NOTHING", parsed.game)
        ids = _player_ids(conn, parsed.players)

        columns = ["pitch_uid", "game_id", "import_id", "pitcher_id", "batter_id",
                   "batter_side", *_PITCH_COLUMNS, "raw"]
        insert = (f"INSERT INTO core.pitch ({', '.join(columns)}) "
                  f"VALUES ({', '.join(['%s'] * len(columns))})")
        with conn.cursor() as cur:
            cur.executemany(insert, [
                (p["pitch_uid"], parsed.game["game_id"], import_id,
                 ids.get(p["pitcher"]), ids.get(p["batter"]), p["batter_side"],
                 *(p[c] for c in _PITCH_COLUMNS), Jsonb(p["raw"]))
                for p in parsed.pitches])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"import_id": import_id, "game_id": parsed.game["game_id"],
            "game_date": parsed.game["game_date"], "pitches": len(parsed.pitches),
            "players": len(parsed.players)}


def _game_pitches(conn, game_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT p.pitch_uid, p.pitch_no, p.rel_speed, p.pitch_call, p.play_result, "
        "       b.team_code, pt.team_code "
        "FROM core.pitch p "
        "LEFT JOIN core.player b ON b.player_id = p.batter_id "
        "LEFT JOIN core.player pt ON pt.player_id = p.pitcher_id "
        "WHERE p.game_id = %s ORDER BY p.pitch_no", (game_id,)).fetchall()
    keys = ("pitch_uid", "pitch_no", "rel_speed", "pitch_call", "play_result",
            "batter_team", "pitcher_team")
    return [dict(zip(keys, r)) for r in rows]


def import_dartfish(conn: psycopg.Connection, path: str) -> dict:
    """Attach a Dartfish file to the TrackMan game it belongs to. The TrackMan
    file for that game must be imported first."""
    tags = dartfish.parse_dartfish(path)
    game_date = next((t["game_date"] for t in tags if t["game_date"]), None)
    try:
        candidates = [r[0] for r in conn.execute(
            "SELECT game_id FROM core.game WHERE game_date = %s ORDER BY game_id", (game_date,))]
        if not candidates:
            raise ImportRefused(f"No TrackMan game is imported for {game_date}. "
                                "Import the TrackMan file first.")

        game_id = pitches = None
        for candidate in candidates:
            rows = _game_pitches(conn, candidate)
            if dartfish.lines_up(tags, rows):
                game_id, pitches = candidate, rows
                break
        if game_id is None:
            counts = ", ".join(f"{c}: {len(_game_pitches(conn, c))}" for c in candidates)
            raise ImportRefused(
                f"The {len(tags)} tagged pitches do not line up with any game on {game_date} "
                f"(TrackMan pitch counts: {counts}). Nothing was imported.")

        import_id = _register_file(conn, "dartfish", path, len(tags))
        if conn.execute("SELECT 1 FROM core.dartfish_pitch d JOIN core.pitch p USING (pitch_uid) "
                        "WHERE p.game_id = %s LIMIT 1", (game_id,)).fetchone():
            raise ImportRefused(f"Game {game_id} already has Dartfish tags.")

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO core.dartfish_pitch (pitch_uid, import_id, baserunning, pitch_result, "
                "hit_result, hit_type, hit_location, hit_xy, hitting_position, well_hit, rbi, raw) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(p["pitch_uid"], import_id, t["baserunning"], t["pitch_result"], t["hit_result"],
                  t["hit_type"], t["hit_location"], t["hit_xy"], t["hitting_position"],
                  t["well_hit"], t["rbi"], Jsonb(t["raw"])) for t, p in zip(tags, pitches)])

        # Dartfish is the only place a team's readable name appears.
        for tag, pitch in zip(tags, pitches):
            for code, name in ((pitch["batter_team"], tag["hitter_team"]),
                               (pitch["pitcher_team"], tag["pitcher_team"])):
                if code and name:
                    conn.execute("UPDATE core.team SET display_name = %s "
                                 "WHERE team_code = %s AND display_name IS NULL", (name, code))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"import_id": import_id, "game_id": game_id, "tags": len(tags),
            "conflicts": dartfish.find_conflicts(tags, pitches)}


def import_file(conn: psycopg.Connection, path: str) -> dict:
    """Work out what kind of file this is and import it."""
    if trackman.is_trackman(path):
        return {"source": "trackman", **import_trackman(conn, path)}
    if dartfish.is_dartfish(path):
        return {"source": "dartfish", **import_dartfish(conn, path)}
    raise ImportRefused("Not a TrackMan or Dartfish file (Yakkertech files are not imported).")


def delete_import(conn: psycopg.Connection, import_id: int) -> None:
    """Remove a file and everything it brought in. A game left with no pitches goes too."""
    conn.execute("DELETE FROM core.import_file WHERE import_id = %s", (import_id,))
    conn.execute("DELETE FROM core.game g WHERE NOT EXISTS "
                 "(SELECT 1 FROM core.pitch p WHERE p.game_id = g.game_id)")
    # Deleting a TrackMan file takes that game's Dartfish tags with it; forget
    # the tag file too, so it can be imported again later.
    conn.execute("DELETE FROM core.import_file f WHERE f.source = 'dartfish' AND NOT EXISTS "
                 "(SELECT 1 FROM core.dartfish_pitch d WHERE d.import_id = f.import_id)")
    conn.commit()
