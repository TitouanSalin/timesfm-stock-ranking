"""Scoring models sharing one interface.

fit() receives the fold's train, validation and refit frames. Any model with
hyperparameters selects them on the validation frame's mean Rank IC over tradable
stocks, then fits its final version on the refit frame. predict() returns one
score per row; higher means more attractive.
"""

from __future__ import annotations

from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .metrics import cross_sectional_ic

TARGET = "y_rank"


def validation_rank_ic(frame: pd.DataFrame, predictions: np.ndarray) -> float:
    scored = frame[["date", "tradable", "y_excess"]].assign(pred=predictions)
    return float(cross_sectional_ic(scored[scored["tradable"]], "pred", "y_excess").mean())


class FactorScore:
    """Scores stocks with a single feature; nothing is estimated."""

    def __init__(self, feature: str, sign: float = 1.0):
        self.feature = feature
        self.sign = sign
        self.selected: dict = {}

    def fit(self, train, val, refit, features) -> "FactorScore":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.sign * frame[self.feature].to_numpy(dtype=float)


class RidgeScore:
    """Ridge regression on ranked features, with missing ranks set to the median."""

    def __init__(self, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7)):
        self.alphas = alphas
        self.selected: dict = {}

    @staticmethod
    def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        return frame[features].fillna(0.0).to_numpy(dtype=np.float64)

    def fit(self, train, val, refit, features) -> "RidgeScore":
        self.features = list(features)
        x_train, y_train = self._matrix(train, self.features), train[TARGET].to_numpy()
        x_val = self._matrix(val, self.features)
        scores = {
            alpha: validation_rank_ic(val, Ridge(alpha=alpha).fit(x_train, y_train).predict(x_val))
            for alpha in self.alphas
        }
        best = max(scores, key=scores.get)
        self.model = Ridge(alpha=best).fit(self._matrix(refit, self.features), refit[TARGET].to_numpy())
        self.selected = {"alpha": best, "val_rank_ic": scores[best]}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._matrix(frame, self.features))


class LightGBMScore:
    """Gradient boosting whose tree parameters and number of rounds are chosen on validation."""

    BASE_PARAMS = {
        "objective": "regression",
        "learning_rate": 0.02,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "deterministic": True,
        "verbosity": -1,
    }
    DEFAULT_GRID = {"num_leaves": (7, 31), "min_data_in_leaf": (500, 5000)}

    def __init__(self, grid=None, rounds=(50, 100, 200, 400), num_threads: int = 32, seed: int = 0):
        self.grid = grid or self.DEFAULT_GRID
        self.rounds = sorted(rounds)
        self.params = {**self.BASE_PARAMS, "num_threads": num_threads, "seed": seed}
        self.selected: dict = {}

    def fit(self, train, val, refit, features) -> "LightGBMScore":
        self.features = list(features)
        train_set = lgb.Dataset(
            train[self.features].to_numpy(dtype=np.float32),
            train[TARGET].to_numpy(),
            free_raw_data=False,
            params={"feature_pre_filter": False},
        )
        x_val = val[self.features].to_numpy(dtype=np.float32)

        best_score, best_params, best_rounds = -np.inf, None, None
        names = list(self.grid)
        for values in product(*self.grid.values()):
            params = dict(zip(names, values))
            booster = lgb.train({**self.params, **params}, train_set, num_boost_round=self.rounds[-1])
            for rounds in self.rounds:
                score = validation_rank_ic(val, booster.predict(x_val, num_iteration=rounds))
                if score > best_score:
                    best_score, best_params, best_rounds = score, params, rounds

        refit_set = lgb.Dataset(refit[self.features].to_numpy(dtype=np.float32), refit[TARGET].to_numpy())
        self.model = lgb.train({**self.params, **best_params}, refit_set, num_boost_round=best_rounds)
        self.selected = {**best_params, "rounds": best_rounds, "val_rank_ic": float(best_score)}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[self.features].to_numpy(dtype=np.float32))


class LightGBMAverage(LightGBMScore):
    """Averages within-date ranks over a fixed grid of tree settings and lengths.

    Nothing is selected, so the validation frame is not used and every model is fitted
    on the refit frame. predict() therefore needs the frame's date column.
    """

    def __init__(self, grid=None, rounds=(100, 200, 400), num_threads: int = 32, seed: int = 0):
        super().__init__(grid=grid, rounds=rounds, num_threads=num_threads, seed=seed)

    def fit(self, train, val, refit, features) -> "LightGBMAverage":
        self.features = list(features)
        refit_set = lgb.Dataset(
            refit[self.features].to_numpy(dtype=np.float32),
            refit[TARGET].to_numpy(),
            free_raw_data=False,
            params={"feature_pre_filter": False},
        )
        names = list(self.grid)
        self.models = [
            lgb.train({**self.params, **dict(zip(names, values))}, refit_set, num_boost_round=self.rounds[-1])
            for values in product(*self.grid.values())
        ]
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy(dtype=np.float32)
        dates = frame["date"].to_numpy()
        ranks = [
            pd.Series(model.predict(x, num_iteration=rounds)).groupby(dates).rank(pct=True)
            for model in self.models
            for rounds in self.rounds
        ]
        return pd.concat(ranks, axis=1).mean(axis=1).to_numpy()
