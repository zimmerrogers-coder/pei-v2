-- PEI tables: everything here is worked out from core.* and can be rebuilt at
-- any time with `python -m app.pei build`. Safe to run more than once.

-- One row per completed plate appearance.
CREATE TABLE IF NOT EXISTS pei.plate_appearance (
    pa_id           SERIAL PRIMARY KEY,
    game_id         TEXT NOT NULL REFERENCES core.game ON DELETE CASCADE,
    inning          SMALLINT NOT NULL,
    top_bottom      TEXT NOT NULL,
    pa_of_inning    SMALLINT NOT NULL,
    batter_id       INT NOT NULL REFERENCES core.player,
    pitcher_id      INT REFERENCES core.player,
    last_pitch_uid  TEXT NOT NULL UNIQUE REFERENCES core.pitch ON DELETE CASCADE,
    n_pitches       SMALLINT NOT NULL,
    outcome         TEXT NOT NULL CHECK (outcome IN
                      ('1B','2B','3B','HR','BB','HBP','K','OUT','FC','ROE','SH','SF')),
    woba_value      REAL NOT NULL,
    is_ab           BOOLEAN NOT NULL,
    is_hit          BOOLEAN NOT NULL,
    -- filled only when the ball was put in play
    in_play         BOOLEAN NOT NULL,
    hit_type        TEXT,
    is_bunt         BOOLEAN NOT NULL DEFAULT false,
    exit_speed      REAL,
    launch_angle    REAL,
    hard_hit        BOOLEAN,        -- NULL when TrackMan did not measure the ball
    sweet_spot      BOOLEAN,
    barrel          BOOLEAN,
    UNIQUE (game_id, inning, top_bottom, pa_of_inning)
);
CREATE INDEX IF NOT EXISTS pa_batter_idx ON pei.plate_appearance (batter_id);

-- One row per hitter per game: counts only, so any rate over any set of games
-- is a sum divided by a sum.
CREATE TABLE IF NOT EXISTS pei.hitter_game (
    game_id        TEXT NOT NULL REFERENCES core.game ON DELETE CASCADE,
    batter_id      INT  NOT NULL REFERENCES core.player,
    pa INT NOT NULL, ab INT NOT NULL, h INT NOT NULL,
    b1 INT NOT NULL, b2 INT NOT NULL, b3 INT NOT NULL, hr INT NOT NULL,
    bb INT NOT NULL, hbp INT NOT NULL, k INT NOT NULL,
    sh INT NOT NULL, sf INT NOT NULL, roe INT NOT NULL,
    woba_num       REAL NOT NULL,   -- sum of woba_value; wOBA = woba_num / pa
    bip            INT NOT NULL,    -- balls in play
    bip_measured   INT NOT NULL,    -- of those, how many have exit speed and angle
    ev_sum         REAL NOT NULL,
    hard_hit INT NOT NULL, sweet_spot INT NOT NULL, barrel INT NOT NULL,
    pitches        INT NOT NULL,
    pitches_located INT NOT NULL,   -- pitches with a plate location
    swings INT NOT NULL, whiffs INT NOT NULL,
    zone_pitches INT NOT NULL, zone_swings INT NOT NULL,
    chase_pitches INT NOT NULL,     -- located pitches outside the zone
    chases INT NOT NULL,            -- swings at those
    PRIMARY KEY (game_id, batter_id)
);
