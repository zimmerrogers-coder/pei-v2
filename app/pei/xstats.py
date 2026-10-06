"""Expected stats: what a batted ball is usually worth, given only how it was hit.

A hitter's results on contact are noisy. A line drive at 72 mph can be caught;
a 45 mph roller can find a hole. xwOBAcon replaces the result of each batted
ball with its *expected* value: the model estimates the chance of an out, a
single, an extra-base hit and a home run from exit speed and launch angle, and
those chances are multiplied by the wOBA weights.

v1 did this with five hand-made buckets and hand-picked values. Here the
relationship is fitted, and the two are compared on balls the model never saw.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.pei.config import Settings

# Four classes. Doubles and triples are one class because there is one triple
# in the data. Everything that is not a hit is an out, including reaching on an
# error or a fielder's choice: the hitter did not earn the base.
CLASSES = ["OUT", "1B", "XB", "HR"]
_CLASS_OF = {"1B": "1B", "2B": "XB", "3B": "XB", "HR": "HR"}

# v1's buckets, exactly as written in player_valuation.py section 8.
V1_BUCKET_VALUE = {"barrel": 1.70, "hard_sweet": 1.30, "hard": 1.00, "sweet": 0.80, "other": 0.25}
_V1_SWEET_SPOT = (10, 22)

MIN_BALLS = 150      # below this, no expected stats are produced at all


def measured(pa: pd.DataFrame) -> pd.DataFrame:
    """The plate appearances that ended on a ball TrackMan measured."""
    return pa[pa["in_play"] & pa["exit_speed"].notna() & pa["launch_angle"].notna()]


def contact_class(outcome: pd.Series) -> pd.Series:
    return outcome.map(_CLASS_OF).fillna("OUT")


def class_values(balls: pd.DataFrame, settings: Settings) -> np.ndarray:
    """wOBA value of each class, in CLASSES order. The extra-base value is the
    average of the doubles and triples it was trained on."""
    w = settings.woba_weights
    extra = balls.loc[balls["outcome"].isin(["2B", "3B"]), "woba_value"]
    return np.array([0.0, w["1B"], float(extra.mean()) if len(extra) else w["2B"], w["HR"]])


def _features(balls: pd.DataFrame) -> np.ndarray:
    return balls[["exit_speed", "launch_angle"]].to_numpy(dtype=float)


def _proba_in_class_order(model, X) -> np.ndarray:
    """Predicted probabilities with one column per entry in CLASSES, even if a
    class was missing from the data the model was fitted on."""
    out = np.zeros((len(X), len(CLASSES)))
    for column, label in enumerate(model.classes_):
        out[:, CLASSES.index(label)] = model.predict_proba(X)[:, column]
    return out


class SmoothModel:
    """Multinomial logistic regression on exit speed, launch angle, their
    squares and their product. Six numbers per class describe the whole
    surface, which is about what 500 batted balls can support. The amount of
    shrinkage on those numbers is chosen by cross-validation."""

    name = "smooth EV/LA model"

    def fit(self, balls: pd.DataFrame, settings: Settings):
        self.values = class_values(balls, settings)
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), cv=5, scoring="neg_log_loss",
                                 max_iter=2000),
        ).fit(_features(balls), contact_class(balls["outcome"]))
        return self

    def proba(self, balls: pd.DataFrame) -> np.ndarray:
        return _proba_in_class_order(self.model, _features(balls))

    def xwobacon(self, balls: pd.DataFrame) -> np.ndarray:
        return self.proba(balls) @ self.values


class FlexibleModel(SmoothModel):
    """Gradient-boosted trees: free to draw any shape. Included only to show
    what happens when a model has more freedom than the data can pay for."""

    name = "flexible model (boosted trees)"

    def fit(self, balls: pd.DataFrame, settings: Settings):
        self.values = class_values(balls, settings)
        self.model = HistGradientBoostingClassifier(random_state=0).fit(
            _features(balls), contact_class(balls["outcome"]))
        return self


def v1_bucket(balls: pd.DataFrame, settings: Settings, sweet_spot=_V1_SWEET_SPOT) -> pd.Series:
    ev, la = balls["exit_speed"], balls["launch_angle"]
    hard = ev >= settings.hard_hit_min_ev
    sweet = la.between(*sweet_spot)
    barrel = (ev >= settings.barrel_min_ev) & la.between(*settings.barrel_la)
    return pd.Series(np.select(
        [barrel, hard & sweet, hard, sweet], ["barrel", "hard_sweet", "hard", "sweet"], "other"),
        index=balls.index)


class FittedBuckets:
    """v1's five buckets, but with each bucket's outcome chances counted from
    the training balls instead of picked by hand. This is the fair version of
    the bucket method: same idea, real numbers."""

    name = "v1 buckets, values fitted"

    def fit(self, balls: pd.DataFrame, settings: Settings):
        self.settings = settings
        self.values = class_values(balls, settings)
        labels = contact_class(balls["outcome"])
        overall = labels.value_counts().reindex(CLASSES, fill_value=0) / len(labels)
        table = pd.crosstab(v1_bucket(balls, settings), labels).reindex(columns=CLASSES, fill_value=0)
        # add one "average" ball to every bucket so no chance is ever exactly 0
        table = table.add(overall, axis=1)
        self.table = table.div(table.sum(axis=1), axis=0)
        self.overall = overall
        return self

    def proba(self, balls: pd.DataFrame) -> np.ndarray:
        rows = [self.table.loc[b] if b in self.table.index else self.overall
                for b in v1_bucket(balls, self.settings)]
        return np.array(rows, dtype=float)

    def xwobacon(self, balls: pd.DataFrame) -> np.ndarray:
        return self.proba(balls) @ self.values


class NoInformation(FittedBuckets):
    """Ignores how the ball was hit: every ball gets the overall outcome chances."""

    name = "no information (league average)"

    def proba(self, balls: pd.DataFrame) -> np.ndarray:
        return np.tile(self.overall.to_numpy(dtype=float), (len(balls), 1))


def evaluate(pa: pd.DataFrame, settings: Settings, repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Score every method on balls it was not fitted on.

    The balls are split into five parts; each method is fitted on four and
    scored on the fifth, for every part, and the whole thing is repeated with
    different splits. Two scores, lower is better for both:

      log_loss  how wrong the outcome chances were
      rmse      how far the expected value was from the ball's actual wOBA value
    """
    balls = measured(pa).reset_index(drop=True)
    labels = contact_class(balls["outcome"])
    methods = [SmoothModel(), FlexibleModel(), FittedBuckets(), NoInformation()]
    scores = {m.name: {"log_loss": [], "sq_err": []} for m in methods}
    scores["v1 buckets, hand-set values"] = {"log_loss": [], "sq_err": []}

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    for train_index, test_index in splitter.split(balls, labels):
        train, test = balls.iloc[train_index], balls.iloc[test_index]
        actual = test["woba_value"].to_numpy(dtype=float)
        for method in methods:
            method.fit(train, settings)
            # scored by class number, so the probability columns stay in CLASSES order
            scores[method.name]["log_loss"].append(log_loss(
                labels.iloc[test_index].map(CLASSES.index), method.proba(test),
                labels=list(range(len(CLASSES)))))
            scores[method.name]["sq_err"].append(np.mean((method.xwobacon(test) - actual) ** 2))
        # v1 as written gives a value, not chances, so it only has an rmse
        hand_set = v1_bucket(test, settings).map(V1_BUCKET_VALUE).to_numpy(dtype=float)
        scores["v1 buckets, hand-set values"]["sq_err"].append(np.mean((hand_set - actual) ** 2))

    rows = []
    for name, s in scores.items():
        rows.append({
            "method": name,
            "log_loss": float(np.mean(s["log_loss"])) if s["log_loss"] else None,
            "rmse": float(np.sqrt(np.mean(s["sq_err"]))),
        })
    return pd.DataFrame(rows).assign(balls=len(balls))


def out_of_fold(pa: pd.DataFrame, settings: Settings) -> pd.Series:
    """xwOBAcon for every measured ball, where each ball's value comes from a
    model fitted on *other games*. A ball never helps predict itself, so a
    hitter's xwOBAcon is not flattered by her own results."""
    balls = measured(pa)
    result = pd.Series(np.nan, index=pa.index, dtype=float)
    if len(balls) < MIN_BALLS or balls["game_id"].nunique() < 2:
        return result                       # too little data to fit anything honest
    folds = GroupKFold(n_splits=min(6, balls["game_id"].nunique()))
    for train_index, test_index in folds.split(balls, groups=balls["game_id"]):
        model = SmoothModel().fit(balls.iloc[train_index], settings)
        result.loc[balls.index[test_index]] = model.xwobacon(balls.iloc[test_index])
    return result
