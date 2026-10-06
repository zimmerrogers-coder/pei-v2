-- Expected stats (Stage 5). Safe to run more than once.

-- Expected wOBA value of the batted ball, from a model fitted on other games.
-- Empty when the PA did not end on a measured ball in play.
ALTER TABLE pei.plate_appearance ADD COLUMN IF NOT EXISTS xwoba_con REAL;

ALTER TABLE pei.hitter_game
    -- sum of xwoba_con over measured balls; xwOBAcon = xwobacon_sum / bip_measured
    ADD COLUMN IF NOT EXISTS xwobacon_sum REAL NOT NULL DEFAULT 0,
    -- like woba_num, but each measured ball counts its expected value instead
    -- of its result; xwOBA = xwoba_num / pa
    ADD COLUMN IF NOT EXISTS xwoba_num REAL NOT NULL DEFAULT 0;
