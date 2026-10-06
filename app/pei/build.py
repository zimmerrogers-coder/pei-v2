"""Rebuilds the PEI tables from the imported pitches.

Nothing in pei.* is edited by hand, so a rebuild simply empties the tables and
fills them again from core.pitch. With 3,000 pitches this takes about a second.
"""

import pandas as pd

from app.pei import config, features, plate_appearances, validate, xstats

_PA_COLUMNS = ["game_id", "inning", "top_bottom", "pa_of_inning", "batter_id", "pitcher_id",
               "last_pitch_uid", "n_pitches", "outcome", "woba_value", "is_ab", "is_hit",
               "in_play", "hit_type", "is_bunt", "exit_speed", "launch_angle",
               "hard_hit", "sweet_spot", "barrel", "xwoba_con"]


def _records(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Rows as plain Python tuples, with every kind of 'missing' turned into None."""
    plain = frame[columns].astype(object).where(frame[columns].notna(), None)
    return [tuple(v.item() if hasattr(v, "item") else v for v in row)
            for row in plain.itertuples(index=False, name=None)]


def _insert(conn, table: str, frame: pd.DataFrame, columns: list[str]) -> None:
    sql = (f"INSERT INTO {table} ({', '.join(columns)}) "
           f"VALUES ({', '.join(['%s'] * len(columns))})")
    with conn.cursor() as cur:
        cur.executemany(sql, _records(frame, columns))


def build(conn, settings: config.Settings | None = None) -> dict:
    settings = settings or config.load()
    pitches = plate_appearances.load_pitches(conn)
    pa, problems = plate_appearances.derive(pitches, settings)
    pa["xwoba_con"] = xstats.out_of_fold(pa, settings)
    games = features.hitter_game(pa, pitches, settings)
    checks = validate.check_all(pa, games)

    conn.execute("TRUNCATE pei.hitter_game, pei.plate_appearance RESTART IDENTITY")
    _insert(conn, "pei.plate_appearance", pa, _PA_COLUMNS)
    _insert(conn, "pei.hitter_game", games, ["game_id", "batter_id", *features.COUNT_COLUMNS])
    conn.commit()
    return {"pitches": len(pitches), "plate_appearances": len(pa), "hitter_games": len(games),
            "hitters": games["batter_id"].nunique(), "problems": problems, "checks": checks}


def load_hitter_games(conn) -> pd.DataFrame:
    """Per-game counts with the game date and the hitter's name and team attached."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT g.game_date, g.season, pl.full_name, pl.team_code, hg.* "
            "FROM pei.hitter_game hg "
            "JOIN core.game g USING (game_id) "
            "JOIN core.player pl ON pl.player_id = hg.batter_id "
            "ORDER BY g.game_date, hg.game_id, hg.batter_id")
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
