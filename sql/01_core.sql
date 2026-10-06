-- Shared tables: what every tab reads. Safe to run more than once.
--
--   core  = data as imported (one row per file, game, player, pitch)
--   pei   = everything the PEI model derives from it (added in later stages)

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS pei;

-- One row per imported file. The hash is what refuses a second import of the
-- same file, even if it was renamed ("... (1).csv").
CREATE TABLE IF NOT EXISTS core.import_file (
    import_id    SERIAL PRIMARY KEY,
    source       TEXT NOT NULL CHECK (source IN ('trackman', 'dartfish')),
    filename     TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    row_count    INT  NOT NULL,
    imported_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, file_hash)
);

-- TrackMan names teams by code (e.g. HOME_SB); Dartfish by display name.
CREATE TABLE IF NOT EXISTS core.team (
    team_code     TEXT PRIMARY KEY,
    display_name  TEXT
);

CREATE TABLE IF NOT EXISTS core.player (
    player_id  SERIAL PRIMARY KEY,
    full_name  TEXT NOT NULL,              -- "First Last"
    team_code  TEXT REFERENCES core.team,
    bats       CHAR(1),
    UNIQUE (full_name, team_code)
);

CREATE TABLE IF NOT EXISTS core.game (
    game_id    TEXT PRIMARY KEY,           -- TrackMan GameID
    game_date  DATE NOT NULL,
    season     INT  NOT NULL,
    home_team  TEXT REFERENCES core.team,
    away_team  TEXT REFERENCES core.team,
    stadium    TEXT
);

-- One row per TrackMan pitch. The columns PEI needs are typed; the whole
-- original row is kept in `raw` so other tabs (the 3D viewer's trajectory
-- numbers) can read fields that have no column here.
CREATE TABLE IF NOT EXISTS core.pitch (
    pitch_uid          TEXT PRIMARY KEY,   -- TrackMan PitchUID
    game_id            TEXT NOT NULL REFERENCES core.game,
    import_id          INT  NOT NULL REFERENCES core.import_file ON DELETE CASCADE,
    pitch_no           INT  NOT NULL,      -- order within the game
    inning             SMALLINT,
    top_bottom         TEXT,
    pa_of_inning       SMALLINT,
    pitch_of_pa        SMALLINT,
    pitcher_id         INT REFERENCES core.player,
    batter_id          INT REFERENCES core.player,
    batter_side        CHAR(1),
    outs               SMALLINT,
    balls              SMALLINT,
    strikes            SMALLINT,
    tagged_pitch_type  TEXT,
    pitch_call         TEXT,
    kor_bb             TEXT,
    play_result        TEXT,
    tagged_hit_type    TEXT,
    outs_on_play       SMALLINT,
    runs_scored        SMALLINT,
    rel_speed          REAL,
    spin_rate          REAL,
    plate_loc_height   REAL,
    plate_loc_side     REAL,
    exit_speed         REAL,
    launch_angle       REAL,
    direction          REAL,
    distance           REAL,
    raw                JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS pitch_game_idx   ON core.pitch (game_id, pitch_no);
CREATE INDEX IF NOT EXISTS pitch_batter_idx ON core.pitch (batter_id);

-- What the Dartfish video tags add to a pitch. TrackMan stays the source for
-- the outcome and the measurements; this is only the extra context.
CREATE TABLE IF NOT EXISTS core.dartfish_pitch (
    pitch_uid         TEXT PRIMARY KEY REFERENCES core.pitch ON DELETE CASCADE,
    import_id         INT  NOT NULL REFERENCES core.import_file ON DELETE CASCADE,
    baserunning       TEXT,                -- "Bases Empty", "R1", "R23", ...
    pitch_result      TEXT,
    hit_result        TEXT,
    hit_type          TEXT,
    hit_location      TEXT,
    hit_xy            TEXT,
    hitting_position  TEXT,                -- "Bunt", "Slap"
    well_hit          BOOLEAN NOT NULL DEFAULT false,
    rbi               SMALLINT,
    raw               JSONB NOT NULL
);
