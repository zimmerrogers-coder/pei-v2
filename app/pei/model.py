"""The PEI v2 model: a hierarchical Bayesian projection of each hitter's wOBA.

A plate appearance is broken into the steps that decide its value:

    strikeout?  --no-->  walk?  --no-->  hit by pitch?  --no-->  ball in play
       p_k                q_bb              p_hbp                 worth `contact`

Each hitter has her own strikeout rate, her own walk rate (among the trips that
are not strikeouts) and her own contact quality (the average expected wOBA of
her batted balls). Each of those three is drawn from a league-wide distribution
whose centre and spread are learned from all hitters at once. That is what
"hierarchical" means, and it is what pulls a 10-PA hitter toward the league
while leaving a 300-PA hitter nearly alone.

The pieces are then put back together:

    wOBA = P(walk) x w_BB + P(HBP) x w_HBP + P(in play) x contact

Because the model is sampled, every hitter gets a few thousand plausible wOBA
values, not one. Their spread is the uncertainty.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm

from app.pei.config import Settings

INTERVAL = (10, 90)        # percentiles of the 80% interval

# Parameters that describe the league rather than one hitter.
LEAGUE_PARAMETERS = ["mu_k", "sigma_k", "mu_bb", "sigma_bb", "p_hbp",
                     "mu_c", "sigma_c", "alpha", "league_woba"]


@dataclass
class ModelData:
    """Everything the model needs, as plain arrays in one fixed hitter order."""
    batter_ids: np.ndarray
    pa: np.ndarray
    k: np.ndarray
    bb: np.ndarray
    hbp: np.ndarray
    ball_hitter: np.ndarray       # for each measured ball, the row of its hitter
    ball_value: np.ndarray        # for each measured ball, its xwOBAcon
    woba_num: np.ndarray          # actual wOBA numerator, for comparison only
    # Balls in play TrackMan did not measure (mostly bunts and pop-ups) have no
    # xwOBAcon. They are handled at league level: what share of balls in play
    # they are, and what they were actually worth on average.
    unmeasured_share: float = 0.0
    unmeasured_value: float = 0.0
    features: dict = field(default_factory=dict)   # component -> (names, matrix)

    @property
    def n_hitters(self) -> int:
        return len(self.batter_ids)


def prepare(pa: pd.DataFrame, features: dict | None = None) -> ModelData:
    """Collapse plate appearances into per-hitter counts and a list of batted
    balls. `pa` needs batter_id, outcome, woba_value and xwoba_con.

    `features` is optional: {"k": frame, "bb": frame, "contact": frame}, each
    indexed by batter_id with one column per skill feature. They are z-scored
    here; a hitter missing a feature gets the average (0)."""
    grouped = pa.groupby("batter_id")
    ids = np.array(sorted(grouped.groups))
    counts = pd.DataFrame({
        "pa": grouped.size(),
        "k": grouped["outcome"].apply(lambda o: (o == "K").sum()),
        "bb": grouped["outcome"].apply(lambda o: (o == "BB").sum()),
        "hbp": grouped["outcome"].apply(lambda o: (o == "HBP").sum()),
        "woba_num": grouped["woba_value"].sum(),
    }).loc[ids]

    balls = pa[pa["xwoba_con"].notna()]
    in_play = ~pa["outcome"].isin(["K", "BB", "HBP"])
    unmeasured = pa[in_play & pa["xwoba_con"].isna()]
    row_of = {batter: row for row, batter in enumerate(ids)}

    prepared = {}
    for component, frame in (features or {}).items():
        matrix = frame.reindex(ids).astype(float)
        matrix = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
        prepared[component] = (list(matrix.columns), matrix.fillna(0.0).to_numpy())

    return ModelData(
        batter_ids=ids,
        pa=counts["pa"].to_numpy(), k=counts["k"].to_numpy(),
        bb=counts["bb"].to_numpy(), hbp=counts["hbp"].to_numpy(),
        ball_hitter=balls["batter_id"].map(row_of).to_numpy(dtype=int),
        ball_value=balls["xwoba_con"].to_numpy(dtype=float),
        woba_num=counts["woba_num"].to_numpy(dtype=float),
        unmeasured_share=len(unmeasured) / max(int(in_play.sum()), 1),
        unmeasured_value=float(unmeasured["woba_value"].mean()) if len(unmeasured) else 0.0,
        features=prepared,
    )


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def build_model(data: ModelData, settings: Settings) -> pm.Model:
    w = settings.woba_weights
    coords = {"hitter": data.batter_ids}
    for component, (names, _) in data.features.items():
        coords[f"{component}_feature"] = names

    def skill(name, centre, centre_sd, spread_sd):
        """centre + (optional features) + spread x z, one value per hitter.
        Written with a standard-normal z ("non-centred") so the sampler copes
        when hitters have very little data."""
        mu = pm.Normal(f"mu_{name}", centre, centre_sd)
        sigma = pm.HalfNormal(f"sigma_{name}", spread_sd)
        z = pm.Normal(f"z_{name}", 0.0, 1.0, dims="hitter")
        value = mu + sigma * z
        component = {"c": "contact"}.get(name, name)
        if component in data.features:
            _, matrix = data.features[component]
            beta = pm.Normal(f"beta_{name}", 0.0, 0.5, dims=f"{component}_feature")
            value = value + pm.math.dot(matrix, beta)
        return mu, value

    with pm.Model(coords=coords) as model:
        # --- strikeouts: of all plate appearances
        mu_k, eta_k = skill("k", _logit(0.15), 1.0, 1.0)
        p_k = pm.Deterministic("p_k", pm.math.invlogit(eta_k), dims="hitter")
        pm.Binomial("k_obs", n=data.pa, p=p_k, observed=data.k)

        # --- walks: of the plate appearances that were not strikeouts
        mu_bb, eta_bb = skill("bb", _logit(0.15), 1.0, 1.0)
        q_bb = pm.Deterministic("q_bb", pm.math.invlogit(eta_bb), dims="hitter")
        pm.Binomial("bb_obs", n=data.pa - data.k, p=q_bb, observed=data.bb)

        # --- hit by pitch: too rare to tell hitters apart, so one league rate
        p_hbp = pm.Beta("p_hbp", 1.0, 30.0)
        pm.Binomial("hbp_obs", n=int((data.pa - data.k - data.bb).sum()), p=p_hbp,
                    observed=int(data.hbp.sum()))

        # --- contact quality: the hitter's average xwOBAcon. Batted-ball values
        # are positive and lopsided (many weak, a few crushed), which is the
        # shape of a Gamma distribution. `alpha` sets how lopsided. (A Beta
        # distribution bounded at a home run's value was tried and rejected: it
        # put the league's average ball at .41 when the data says .375.)
        mu_c, eta_c = skill("c", float(np.log(0.37)), 0.5, 0.5)
        contact = pm.Deterministic("contact", pm.math.exp(eta_c), dims="hitter")
        alpha = pm.Gamma("alpha", 2.0, 1.0)
        pm.Gamma("ball_obs", alpha=alpha, beta=alpha / contact[data.ball_hitter],
                 observed=np.maximum(data.ball_value, 1e-4))

        # --- put the plate appearance back together
        def woba_of(pk, qbb, con):
            reach = 1 - pk
            in_play_value = ((1 - data.unmeasured_share) * con
                             + data.unmeasured_share * data.unmeasured_value)
            return (reach * qbb * w["BB"]
                    + reach * (1 - qbb) * p_hbp * w["HBP"]
                    + reach * (1 - qbb) * (1 - p_hbp) * in_play_value)

        pm.Deterministic("woba", woba_of(p_k, q_bb, contact), dims="hitter")
        # a hitter exactly at the centre of all three league distributions
        pm.Deterministic("league_woba", woba_of(
            pm.math.invlogit(mu_k), pm.math.invlogit(mu_bb), pm.math.exp(mu_c)))
    return model


def fit(data: ModelData, settings: Settings, draws: int = 1500, tune: int = 1500,
        chains: int = 4, seed: int = 20260313, progressbar: bool = False):
    """Sample the model. Returns (model, trace)."""
    model = build_model(data, settings)
    with model:
        trace = pm.sample(draws, tune=tune, chains=chains, target_accept=0.95,
                          random_seed=seed, progressbar=progressbar)
    return model, trace


def _draws(trace, name: str) -> np.ndarray:
    """All posterior draws of a variable, chains stacked: (draws,) or (draws, hitters)."""
    values = np.asarray(trace.posterior[name].values)
    return values.reshape(-1, *values.shape[2:])


def diagnostics(trace) -> dict:
    """The three things to look at before believing a fit.

    divergences  times the sampler lost its footing; must be 0
    max_rhat     do the separate chains agree? must be under 1.01
    min_ess      how many independent draws the result is worth; want > 400
    """
    import arviz as az

    names = [n for n in trace.posterior.data_vars
             if not n.startswith("z_")]          # z's are internal scaffolding
    rhat, ess = az.rhat(trace, var_names=names), az.ess(trace, var_names=names)

    def extreme(result, fn):
        dataset = result.to_dataset() if hasattr(result, "to_dataset") else result
        return float(fn([float(fn(np.asarray(dataset[v].values))) for v in dataset.data_vars]))

    return {
        "divergences": int(np.asarray(trace.sample_stats["diverging"].values).sum()),
        "max_rhat": extreme(rhat, np.nanmax),
        "min_ess": extreme(ess, np.nanmin),
        "draws": int(_draws(trace, "p_hbp").shape[0]),
    }


def hitter_summary(trace, data: ModelData) -> pd.DataFrame:
    """One row per hitter: the projection and its 80% interval, for wOBA and
    for each of the three skills behind it."""
    out = pd.DataFrame({
        "batter_id": data.batter_ids,
        "n_pa": data.pa,
        "n_balls": np.bincount(data.ball_hitter, minlength=data.n_hitters),
        "raw_woba": data.woba_num / data.pa,
        "raw_k_rate": data.k / data.pa,
        "raw_bb_rate": data.bb / data.pa,
    })
    walk = (1 - _draws(trace, "p_k")) * _draws(trace, "q_bb")      # walks per PA
    for column, draws in (("woba", _draws(trace, "woba")), ("k_rate", _draws(trace, "p_k")),
                          ("bb_rate", walk), ("contact", _draws(trace, "contact"))):
        low, high = np.percentile(draws, INTERVAL, axis=0)
        out[f"proj_{column}"] = draws.mean(axis=0)
        out[f"{column}_lo"] = low
        out[f"{column}_hi"] = high
    return out


def league_summary(trace, data: ModelData) -> pd.DataFrame:
    """The league-level parameters, plus any feature weights."""
    rows = []
    names = list(LEAGUE_PARAMETERS)
    for name in names:
        draws = _draws(trace, name)
        low, high = np.percentile(draws, INTERVAL)
        rows.append({"parameter": name, "mean": float(draws.mean()),
                     "lo_80": float(low), "hi_80": float(high)})
    for component, (feature_names, _) in data.features.items():
        short = {"contact": "c"}.get(component, component)
        draws = _draws(trace, f"beta_{short}")
        for column, feature in enumerate(feature_names):
            low, high = np.percentile(draws[:, column], INTERVAL)
            rows.append({"parameter": f"weight: {feature} -> {component}",
                         "mean": float(draws[:, column].mean()),
                         "lo_80": float(low), "hi_80": float(high)})
    return pd.DataFrame(rows)


def simulate(n_hitters: int = 66, seed: int = 11, settings: Settings | None = None,
             truth: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Invent hitters whose true skills are known, then invent their plate
    appearances. Returns (pa, truth) where `truth` has each hitter's real wOBA.

    The sample sizes copy the real data: a few regulars with 20 to 45 PA and
    many hitters with a handful."""
    from app.pei import config
    settings = settings or config.load()
    w = settings.woba_weights
    t = {"mu_k": _logit(0.14), "sigma_k": 0.6, "mu_bb": _logit(0.16), "sigma_bb": 0.4,
         "p_hbp": 0.02, "mu_c": float(np.log(0.37)), "sigma_c": 0.25, "alpha": 1.2,
         **(truth or {})}
    rng = np.random.default_rng(seed)

    n_pa = np.where(rng.random(n_hitters) < 0.2,
                    rng.integers(18, 46, n_hitters), rng.integers(2, 14, n_hitters))
    p_k = 1 / (1 + np.exp(-(t["mu_k"] + t["sigma_k"] * rng.standard_normal(n_hitters))))
    q_bb = 1 / (1 + np.exp(-(t["mu_bb"] + t["sigma_bb"] * rng.standard_normal(n_hitters))))
    contact = np.exp(t["mu_c"] + t["sigma_c"] * rng.standard_normal(n_hitters))

    rows = []
    for hitter in range(n_hitters):
        for _ in range(n_pa[hitter]):
            if rng.random() < p_k[hitter]:
                rows.append((hitter, "K", 0.0, np.nan))
            elif rng.random() < q_bb[hitter]:
                rows.append((hitter, "BB", w["BB"], np.nan))
            elif rng.random() < t["p_hbp"]:
                rows.append((hitter, "HBP", w["HBP"], np.nan))
            else:
                value = rng.gamma(t["alpha"], contact[hitter] / t["alpha"])
                rows.append((hitter, "OUT", value, value))
    pa = pd.DataFrame(rows, columns=["batter_id", "outcome", "woba_value", "xwoba_con"])

    reach = 1 - p_k
    true_woba = (reach * q_bb * w["BB"] + reach * (1 - q_bb) * t["p_hbp"] * w["HBP"]
                 + reach * (1 - q_bb) * (1 - t["p_hbp"]) * contact)
    return pa, pd.DataFrame({"batter_id": np.arange(n_hitters), "true_woba": true_woba,
                             "true_k_rate": p_k, "true_contact": contact, "n_pa": n_pa})
