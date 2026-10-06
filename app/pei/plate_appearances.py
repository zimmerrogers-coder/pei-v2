"""Turns pitches into plate appearances.

TrackMan has one row per pitch. A plate appearance (PA) is the run of pitches
that share a game, inning, half-inning and PAofInning, and it ends on the pitch
that is a strikeout, a walk, a hit-by-pitch or a ball in play. The outcome
mapping follows PEI v1 (player_valuation.py) with three changes agreed for v2:

  * "Error" is its own outcome, reached on error: an at-bat, not a hit, worth 0
  * "Sacrifice" is split into a bunt (SH) or a fly (SF) using the hit type,
    with the Dartfish bunt tag as a second source
  * every rate uses plate appearances as its denominator, so nothing here
    depends on what counts as an at-bat
"""

import pandas as pd

from app.pei.config import Settings

PA_KEY = ["game_id", "inning", "top_bottom", "pa_of_inning"]

_PLAY_RESULT = {
    "Single": "1B", "Double": "2B", "Triple": "3B", "HomeRun": "HR",
    "Out": "OUT", "FieldersChoice": "FC", "Error": "ROE",
}
HITS = {"1B", "2B", "3B", "HR"}
NOT_AT_BAT = {"BB", "HBP", "SH", "SF"}

PITCH_QUERY = """
    SELECT p.pitch_uid, p.game_id, p.pitch_no, p.inning, p.top_bottom, p.pa_of_inning,
           p.batter_id, p.pitcher_id, p.balls, p.strikes, p.pitch_call, p.kor_bb,
           p.play_result, p.tagged_hit_type, p.exit_speed, p.launch_angle,
           p.plate_loc_height, p.plate_loc_side, d.hitting_position
    FROM core.pitch p
    LEFT JOIN core.dartfish_pitch d USING (pitch_uid)
    ORDER BY p.game_id, p.pitch_no
"""


def load_pitches(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(PITCH_QUERY)
        columns = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def ends_pa(pitches: pd.DataFrame) -> pd.Series:
    return (pitches["kor_bb"].isin(["Strikeout", "Walk"])
            | pitches["pitch_call"].isin(["InPlay", "HitByPitch"]))


def _outcome(row) -> str | None:
    """The outcome code for the pitch that ended a PA, or None if unrecognised."""
    if row.pitch_call == "HitByPitch":          # checked first: TrackMan sometimes
        return "HBP"                            # also marks these as a walk
    if row.pitch_call == "InPlay":
        if row.play_result == "Sacrifice":
            bunt = row.tagged_hit_type == "Bunt" or row.hitting_position == "Bunt"
            return "SH" if bunt else "SF"
        return _PLAY_RESULT.get(row.play_result)
    if row.kor_bb == "Strikeout":
        return "K"
    if row.kor_bb == "Walk":
        return "BB"
    return None


def _credited_batter(group: pd.DataFrame, outcome: str) -> int:
    """Who the plate appearance belongs to. Normally the only batter. When a
    pinch hitter takes over mid-count, it is the pinch hitter's, with the
    scoring-rule exception that a strikeout goes to the batter who left if she
    left with two strikes already on her."""
    batters = group["batter_id"].tolist()
    final = batters[-1]
    if outcome == "K" and batters[0] != final:
        entered = batters.index(final)
        if group["strikes"].iloc[entered] == 2:
            return int(batters[entered - 1])
    return int(final)


def derive(pitches: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (plate_appearances, problems). `problems` lists every group of
    pitches that could not be turned into a plate appearance, and why."""
    pitches = pitches.sort_values(["game_id", "pitch_no"]).copy()
    pitches["ends"] = ends_pa(pitches)

    kept, problems = [], []
    for key, group in pitches.groupby(PA_KEY, sort=False):
        last = group.iloc[-1]
        reason = None
        if group["ends"].sum() == 0:
            reason = "no ending pitch (the half-inning ended during the plate appearance)"
        elif group["ends"].sum() > 1 or not last.ends:
            reason = "more than one ending pitch, or pitches after the ending pitch"
        elif _outcome(last) is None:
            reason = f"unrecognised result {last.play_result!r}"
        if reason:
            problems.append({**dict(zip(PA_KEY, key)), "pitches": len(group), "reason": reason})
            continue
        kept.append({
            **dict(zip(PA_KEY, key)),
            "batter_id": _credited_batter(group, _outcome(last)),
            "pitcher_id": None if pd.isna(last.pitcher_id) else int(last.pitcher_id),
            "last_pitch_uid": last.pitch_uid,
            "n_pitches": len(group),
            "outcome": _outcome(last),
            "in_play": last.pitch_call == "InPlay",
            "hit_type": last.tagged_hit_type if last.pitch_call == "InPlay" else None,
            "is_bunt": bool(last.pitch_call == "InPlay"
                            and (last.tagged_hit_type == "Bunt" or last.hitting_position == "Bunt")),
            "exit_speed": last.exit_speed if last.pitch_call == "InPlay" else None,
            "launch_angle": last.launch_angle if last.pitch_call == "InPlay" else None,
            "balls": last.balls, "strikes": last.strikes,
        })

    pa = pd.DataFrame(kept)
    pa["woba_value"] = pa["outcome"].map(settings.woba_weights).fillna(0.0)
    pa["is_hit"] = pa["outcome"].isin(HITS)
    pa["is_ab"] = ~pa["outcome"].isin(NOT_AT_BAT)

    # Contact quality only exists where TrackMan measured the ball; an unmeasured
    # ball is "unknown" (None), not "not hard hit".
    measured = pa["exit_speed"].notna() & pa["launch_angle"].notna()
    ev, la = pa["exit_speed"], pa["launch_angle"]
    flags = {
        "hard_hit": ev >= settings.hard_hit_min_ev,
        "sweet_spot": la.between(*settings.sweet_spot_la),
        "barrel": (ev >= settings.barrel_min_ev) & la.between(*settings.barrel_la),
    }
    for name, flag in flags.items():
        pa[name] = flag.astype("boolean").where(measured, pd.NA)
    pa["xwoba_con"] = float("nan")              # filled by xstats during a build
    return pa, pd.DataFrame(problems, columns=[*PA_KEY, "pitches", "reason"])
