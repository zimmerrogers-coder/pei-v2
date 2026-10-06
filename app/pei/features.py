"""Per-hitter counts and the rates built from them.

Counts are stored per hitter per game. A rate is never stored: it is always a
sum of counts divided by a sum of counts, over whichever games are asked for.
That is what lets the same table answer "season to date", "first six games"
and "everything after April 1" without recomputing anything.
"""

import pandas as pd

from app.pei.config import Settings

SWINGS = {"StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable", "FoulBall", "InPlay"}

COUNT_COLUMNS = [
    "pa", "ab", "h", "b1", "b2", "b3", "hr", "bb", "hbp", "k", "sh", "sf", "roe",
    "woba_num", "bip", "bip_measured", "ev_sum", "hard_hit", "sweet_spot", "barrel",
    "pitches", "pitches_located", "swings", "whiffs",
    "zone_pitches", "zone_swings", "chase_pitches", "chases",
    "xwobacon_sum", "xwoba_num",
]
_FRACTIONAL = ("woba_num", "ev_sum", "xwobacon_sum", "xwoba_num")

# rate -> (numerator, denominator). Plate appearances are the denominator for
# every outcome rate; contact rates use measured balls in play.
RATES = {
    "woba": ("woba_num", "pa"),
    "k_rate": ("k", "pa"),
    "bb_rate": ("bb", "pa"),
    "hr_rate": ("hr", "pa"),
    "avg_ev": ("ev_sum", "bip_measured"),
    "hard_hit_rate": ("hard_hit", "bip_measured"),
    "sweet_spot_rate": ("sweet_spot", "bip_measured"),
    "barrel_rate": ("barrel", "bip_measured"),
    "swing_rate": ("swings", "pitches"),
    "whiff_rate": ("whiffs", "swings"),
    "zone_swing_rate": ("zone_swings", "zone_pitches"),
    "chase_rate": ("chases", "chase_pitches"),
    "xwobacon": ("xwobacon_sum", "bip_measured"),
    "xwoba": ("xwoba_num", "pa"),
}


def pitch_flags(pitches: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Adds swing / whiff / in-zone flags to every pitch."""
    out = pitches.copy()
    out["located"] = out["plate_loc_height"].notna() & out["plate_loc_side"].notna()
    low, high = settings.zone_height
    out["in_zone"] = (out["located"]
                      & out["plate_loc_height"].between(low, high)
                      & (out["plate_loc_side"].abs() <= settings.zone_half_width))
    out["swing"] = out["pitch_call"].isin(SWINGS)
    out["whiff"] = out["pitch_call"] == "StrikeSwinging"
    return out


def hitter_game(pa: pd.DataFrame, pitches: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row of counts per (game, hitter)."""
    key = ["game_id", "batter_id"]
    o = pa["outcome"]
    measured = pa["hard_hit"].notna()
    from_pa = pd.DataFrame({
        "game_id": pa["game_id"], "batter_id": pa["batter_id"],
        "pa": 1, "ab": pa["is_ab"], "h": pa["is_hit"],
        "b1": o == "1B", "b2": o == "2B", "b3": o == "3B", "hr": o == "HR",
        "bb": o == "BB", "hbp": o == "HBP", "k": o == "K",
        "sh": o == "SH", "sf": o == "SF", "roe": o == "ROE",
        "woba_num": pa["woba_value"],
        "bip": pa["in_play"], "bip_measured": measured,
        "ev_sum": pa["exit_speed"].where(measured, 0.0).astype(float),
        "hard_hit": pa["hard_hit"].fillna(False), "sweet_spot": pa["sweet_spot"].fillna(False),
        "barrel": pa["barrel"].fillna(False),
        "xwobacon_sum": pa["xwoba_con"].fillna(0.0),
        # expected value where the ball was measured, the actual result otherwise
        "xwoba_num": pa["xwoba_con"].fillna(pa["woba_value"]),
    }).astype({c: int for c in ("ab", "h", "b1", "b2", "b3", "hr", "bb", "hbp", "k", "sh", "sf",
                                "roe", "bip", "bip_measured", "hard_hit", "sweet_spot", "barrel")})
    from_pa = from_pa.groupby(key, as_index=False).sum()

    # Only pitches that belong to a kept plate appearance are counted, so the
    # pitch-level and PA-level numbers describe the same trips to the plate.
    kept = pitches.merge(pa[["game_id", "inning", "top_bottom", "pa_of_inning"]], how="inner")
    f = pitch_flags(kept, settings)
    out_of_zone = f["located"] & ~f["in_zone"]
    from_pitches = pd.DataFrame({
        "game_id": f["game_id"], "batter_id": f["batter_id"].astype(int),
        "pitches": 1, "pitches_located": f["located"],
        "swings": f["swing"], "whiffs": f["whiff"],
        "zone_pitches": f["in_zone"], "zone_swings": f["in_zone"] & f["swing"],
        "chase_pitches": out_of_zone, "chases": out_of_zone & f["swing"],
    }).astype({c: int for c in ("pitches_located", "swings", "whiffs", "zone_pitches",
                                "zone_swings", "chase_pitches", "chases")})
    from_pitches = from_pitches.groupby(key, as_index=False).sum()

    # outer: a pinch hitter can see pitches in a game without finishing a PA
    merged = from_pa.merge(from_pitches, on=key, how="outer").fillna(0)
    whole = [c for c in COUNT_COLUMNS if c not in _FRACTIONAL]
    return merged.astype({c: int for c in whole})[key + COUNT_COLUMNS]


def rates(counts: pd.DataFrame) -> pd.DataFrame:
    """Adds every rate in RATES to a table of counts. A rate with a zero
    denominator is left empty (NaN), never 0."""
    out = counts.copy()
    for name, (numerator, denominator) in RATES.items():
        bottom = out[denominator].where(out[denominator] > 0)
        out[name] = out[numerator] / bottom
    return out


def hitter_totals(games: pd.DataFrame) -> pd.DataFrame:
    """Sum per-game counts to one row per hitter, with rates."""
    return rates(games.groupby("batter_id", as_index=False)[COUNT_COLUMNS].sum())
