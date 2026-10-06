-- Holdout evaluation (Stage 8). Safe to run more than once.
-- Filled by `python -m app.pei evaluate --version <name>`.

-- Every method's score on held-out games, pooled over the splits of a scheme.
CREATE TABLE IF NOT EXISTS pei.evaluation (
    version      TEXT NOT NULL,
    scheme       TEXT NOT NULL,        -- 'chronological' or 'random halves'
    method       TEXT NOT NULL,
    hitters      INT  NOT NULL,        -- hitter-splits scored (seen in training)
    holdout_pa   INT  NOT NULL,
    rmse         REAL NOT NULL,        -- weighted by held-out plate appearances
    mae          REAL NOT NULL,
    bias         REAL NOT NULL,        -- predicted minus actual; negative = predicted too low
    rank_corr    REAL,
    coverage_80  REAL,                 -- share of actual results inside the 80% interval
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (version, scheme, method)
);

-- The model against each other method: how much lower its error is, and how
-- often that holds when the held-out hitters are resampled.
CREATE TABLE IF NOT EXISTS pei.evaluation_comparison (
    version       TEXT NOT NULL,
    scheme        TEXT NOT NULL,
    method        TEXT NOT NULL,
    versus        TEXT NOT NULL,
    rmse_gain     REAL NOT NULL,       -- other's RMSE minus the method's; positive = method better
    gain_lo       REAL NOT NULL,       -- 10th and 90th percentile of that gain
    gain_hi       REAL NOT NULL,
    share_better  REAL NOT NULL,
    PRIMARY KEY (version, scheme, method, versus)
);

ALTER TABLE pei.model_version
    ADD COLUMN IF NOT EXISTS holdout_rmse        REAL,   -- chronological scheme
    ADD COLUMN IF NOT EXISTS best_baseline       TEXT,
    ADD COLUMN IF NOT EXISTS best_baseline_rmse  REAL;
