"""Stage 1 check: PyMC is installed and can fit a small hierarchical model here.

The model is the same *shape* as the real PEI model (Stage 7): each hitter has a
true strikeout rate drawn from a league-wide distribution, and we only see a
handful of plate appearances per hitter. The data is simulated, so we know the
right answer and can check the sampler found it.
"""

import time

import numpy as np
import pymc as pm

N_HITTERS = 67        # hitters in the 12 TrackMan games
LEAGUE_RATE = 0.15    # simulated league strikeout rate
TALENT_SD = 0.5       # spread of true talent, on the log-odds scale


def _simulate(seed=7):
    rng = np.random.default_rng(seed)
    pa = rng.integers(2, 45, size=N_HITTERS)
    league_logit = np.log(LEAGUE_RATE / (1 - LEAGUE_RATE))
    talent = league_logit + TALENT_SD * rng.standard_normal(N_HITTERS)
    strikeouts = rng.binomial(pa, 1 / (1 + np.exp(-talent)))
    return pa, strikeouts


def test_hierarchical_model_samples_cleanly():
    pa, strikeouts = _simulate()

    with pm.Model():
        mu = pm.Normal("mu", -1.5, 1.0)
        sigma = pm.HalfNormal("sigma", 1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=N_HITTERS)
        rate = pm.Deterministic("rate", pm.math.invlogit(mu + sigma * z))
        pm.Binomial("k", n=pa, p=rate, observed=strikeouts)

        start = time.perf_counter()
        trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=7, progressbar=False)
        seconds = time.perf_counter() - start

    league = float((1 / (1 + np.exp(-trace.posterior["mu"].values))).mean())
    divergences = int(trace.sample_stats["diverging"].sum())
    print(f"\nsampled in {seconds:.0f}s, league rate {league:.3f}, divergences {divergences}")

    assert divergences == 0
    assert abs(league - LEAGUE_RATE) < 0.04
    assert seconds < 300, "sampling is too slow on this machine to iterate on a model"
