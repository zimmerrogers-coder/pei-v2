-- Split-half reliability (Stage 6). Safe to run more than once.
-- Filled by `python -m app.pei reliability`; each run replaces the contents.

-- The measured points: reliability of each metric at each sample size.
CREATE TABLE IF NOT EXISTS pei.reliability_curve (
    metric   TEXT NOT NULL,
    n        INT  NOT NULL,       -- events per hitter (PA, measured balls, swings, ...)
    hitters  INT  NOT NULL,       -- hitters with at least n events
    r_half   REAL NOT NULL,       -- average correlation between the two halves
    r        REAL NOT NULL,       -- stepped up to n events with Spearman-Brown
    PRIMARY KEY (metric, n)
);

-- One row per metric: the fitted curve r = n / (n + k).
CREATE TABLE IF NOT EXISTS pei.metric_reliability (
    metric            TEXT PRIMARY KEY,
    denominator       TEXT NOT NULL,      -- what one event is
    league_value      REAL NOT NULL,
    events            INT  NOT NULL,
    hitters           INT  NOT NULL,
    median_events     REAL NOT NULL,      -- per hitter
    max_events        INT  NOT NULL,
    k                 REAL,               -- events at which r = 0.5; NULL = no detectable signal
    k_lo              REAL,               -- 10th and 90th percentile of k when the
    k_hi              REAL,               --   hitters are resampled
    share_no_signal   REAL,               -- share of resamples with no detectable signal
    stabilization_n   REAL,               -- events at which r = 0.70
    largest_n_tested  INT,
    r_at_largest_n    REAL,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
