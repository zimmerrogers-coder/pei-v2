"""Stage 7 checks: the hierarchical model, tested on simulated hitters whose
true talent is known."""

import warnings

import numpy as np
import pandas as pd
import pytest

from app.pei import config, fit, model

SETTINGS = config.load()
warnings.filterwarnings("ignore")


def test_prepare_counts_and_ball_list():
    pa = pd.DataFrame({
        "batter_id": [7, 7, 7, 7, 3, 3],
        "outcome": ["K", "BB", "1B", "OUT", "HBP", "OUT"],
        "woba_value": [0.0, 0.69, 0.88, 0.0, 0.72, 0.0],
        "xwoba_con": [np.nan, np.nan, 0.61, 0.20, np.nan, np.nan],   # last ball unmeasured
    })
    data = model.prepare(pa)
    assert list(data.batter_ids) == [3, 7]
    assert list(data.pa) == [2, 4] and list(data.k) == [0, 1]
    assert list(data.bb) == [0, 1] and list(data.hbp) == [1, 0]
    assert list(data.ball_hitter) == [1, 1] and list(data.ball_value) == [0.61, 0.20]
    assert data.unmeasured_share == pytest.approx(1 / 3)     # 1 of 3 balls in play
    assert data.unmeasured_value == 0.0


def test_features_are_optional_and_standardised():
    pa, _ = model.simulate(n_hitters=12, seed=2, settings=SETTINGS)
    whiff = pd.DataFrame({"whiff_rate": np.linspace(0.1, 0.4, 12)}, index=range(12))
    whiff.loc[5, "whiff_rate"] = np.nan                       # a hitter with no swings
    data = model.prepare(pa, features={"k": whiff})
    names, matrix = data.features["k"]
    assert names == ["whiff_rate"]
    assert matrix[5, 0] == 0.0                                # missing -> league average
    assert abs(np.delete(matrix[:, 0], 5).mean()) < 1e-9

    plain = {v.name for v in model.build_model(model.prepare(pa), SETTINGS).free_RVs}
    with_feature = {v.name for v in model.build_model(data, SETTINGS).free_RVs}
    assert with_feature - plain == {"beta_k"}


@pytest.fixture(scope="module")
def recovered():
    return fit.recovery(seed=11, draws=600, tune=800)


def test_fit_is_clean(recovered):
    d = recovered["diagnostics"]
    assert d["divergences"] == 0
    assert d["max_rhat"] < 1.02            # the short test run; the real run needs < 1.01
    assert d["min_ess"] > 200


def test_model_is_closer_to_the_truth_than_raw_numbers(recovered):
    assert recovered["rmse_model"] < 0.6 * recovered["rmse_raw"]


def test_intervals_contain_the_truth_about_as_often_as_claimed(recovered):
    # 66 hitters: an honest 80% interval lands between roughly 60% and 95%
    assert 0.60 <= recovered["coverage_80"] <= 0.95


def test_small_samples_are_pulled_in_and_have_wider_intervals(recovered):
    table = recovered["table"]
    few, many = table[table["n_pa"] <= 5], table[table["n_pa"] >= 25]
    assert few["proj_woba"].std() < few["raw_woba"].std() / 2      # pulled toward the league
    width = table["woba_hi"] - table["woba_lo"]
    assert width[few.index].mean() > width[many.index].mean()
    assert (table["woba_lo"] < table["proj_woba"]).all()
    assert (table["proj_woba"] < table["woba_hi"]).all()
