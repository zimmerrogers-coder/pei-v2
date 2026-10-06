"""Checks that the derived tables are internally consistent.

Each check returns the rows that break it. An empty result means it passed.
"""

import pandas as pd


def check_all(pa: pd.DataFrame, games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    g = games
    show = ["game_id", "batter_id"]
    results = {
        # every plate appearance is exactly one of the things it can be
        "PA = AB + BB + HBP + SH + SF":
            g[g["pa"] != g["ab"] + g["bb"] + g["hbp"] + g["sh"] + g["sf"]][show],
        "H <= AB": g[g["h"] > g["ab"]][show],
        "H = 1B + 2B + 3B + HR": g[g["h"] != g["b1"] + g["b2"] + g["b3"] + g["hr"]][show],
        "measured balls <= balls in play": g[g["bip_measured"] > g["bip"]][show],
        "whiffs <= swings <= pitches":
            g[(g["whiffs"] > g["swings"]) | (g["swings"] > g["pitches"])][show],
        "zone + chase pitches = located pitches":
            g[g["zone_pitches"] + g["chase_pitches"] != g["pitches_located"]][show],
        # the count before the last pitch must allow the result
        "strikeout came with 2 strikes": pa[(pa["outcome"] == "K") & (pa["strikes"] != 2)],
        "walk came with 3 balls": pa[(pa["outcome"] == "BB") & (pa["balls"] != 3)],
    }

    # game totals: the per-game table must add up to the plate-appearance table
    by_game = pa.groupby("game_id").agg(pa=("outcome", "size"), woba=("woba_value", "sum"))
    summed = g.groupby("game_id").agg(pa=("pa", "sum"), woba=("woba_num", "sum"))
    joined = by_game.join(summed, lsuffix="_pa", rsuffix="_games", how="outer").fillna(0)
    off = (joined["pa_pa"] != joined["pa_games"]) | ((joined["woba_pa"] - joined["woba_games"]).abs() > 1e-6)
    results["game totals = sum of plate appearances"] = joined[off].reset_index()
    return results
