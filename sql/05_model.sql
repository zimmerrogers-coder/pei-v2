-- Model runs and their projections (Stage 7). Safe to run more than once.
-- Filled by `python -m app.pei fit --version <name>`.

-- One row per model run: what it was fitted on and whether the fit is trustworthy.
CREATE TABLE IF NOT EXISTS pei.model_version (
    version            TEXT PRIMARY KEY,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    games              INT  NOT NULL,
    train_through      DATE NOT NULL,       -- date of the last game used
    hitters            INT  NOT NULL,
    plate_appearances  INT  NOT NULL,
    measured_balls     INT  NOT NULL,
    settings           JSONB NOT NULL,      -- wOBA weights and thresholds used
    diagnostics        JSONB NOT NULL,      -- divergences, max r-hat, min effective sample size
    checks             JSONB NOT NULL,      -- model-vs-data checks
    notes              TEXT
);

-- League-level parameters of a run, each with its 80% interval.
CREATE TABLE IF NOT EXISTS pei.model_parameter (
    version    TEXT NOT NULL REFERENCES pei.model_version ON DELETE CASCADE,
    parameter  TEXT NOT NULL,
    mean       REAL NOT NULL,
    lo_80      REAL NOT NULL,
    hi_80      REAL NOT NULL,
    PRIMARY KEY (version, parameter)
);

-- One row per hitter per run. "raw" is what happened; "proj" is the model's
-- estimate of true talent; lo/hi are the 10th and 90th percentile.
CREATE TABLE IF NOT EXISTS pei.projection (
    version       TEXT NOT NULL REFERENCES pei.model_version ON DELETE CASCADE,
    batter_id     INT  NOT NULL REFERENCES core.player,
    n_pa          INT  NOT NULL,
    n_balls       INT  NOT NULL,            -- measured balls in play
    raw_woba      REAL NOT NULL,
    proj_woba     REAL NOT NULL, woba_lo    REAL NOT NULL, woba_hi    REAL NOT NULL,
    raw_k_rate    REAL NOT NULL,
    proj_k_rate   REAL NOT NULL, k_rate_lo  REAL NOT NULL, k_rate_hi  REAL NOT NULL,
    raw_bb_rate   REAL NOT NULL,
    proj_bb_rate  REAL NOT NULL, bb_rate_lo REAL NOT NULL, bb_rate_hi REAL NOT NULL,
    proj_contact  REAL NOT NULL, contact_lo REAL NOT NULL, contact_hi REAL NOT NULL,
    PRIMARY KEY (version, batter_id)
);
