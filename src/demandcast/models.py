"""Baselines and a direct multi-horizon tree model with fixed iteration count."""

from contextlib import nullcontext

import numpy as np
import pandas as pd

from demandcast.features import OriginFeatures


class Baseline:
    def __init__(self, name: str):
        self.name = name

    def fit(self, panel):
        if len(panel.dates) < 28:
            raise ValueError("Baseline needs 28 observed days")
        self.last28 = panel.values[:, -28:].copy()
        return self

    def predict(self, horizon: int):
        if self.name == "seasonal_naive":
            return np.tile(self.last28[:, -7:], (1, (horizon + 6) // 7))[:, :horizon]
        return np.repeat(self.last28.mean(axis=1, keepdims=True), horizon, axis=1)


class DirectTree:
    def __init__(self, name, config):
        self.name, self.config = name, config

    def _thread_context(self):
        if self.name == "histgbm":
            from threadpoolctl import threadpool_limits
            return threadpool_limits(limits=self.config.threads)
        return nullcontext()

    def fit(self, panel):
        self.features = OriginFeatures(panel)
        x, y, self.training_origins = self.features.training_matrix(
            self.config.horizon, self.config.training_origin_days, self.config.origin_stride)
        self.constant = float(y[0]) if np.all(y == y[0]) else None
        if self.constant is not None:
            return self
        if self.name == "histgbm":
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
            except ImportError as exc:
                raise ImportError('Install ML dependencies: pip install -e ".[ml]"') from exc
            if any(len(mapping) > 255 for mapping in self.features.mappings.values()):
                raise ValueError("HistGBM supports at most 255 categories per static feature")
            self.estimator = HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.05, max_iter=self.config.iterations,
                max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False,
                categorical_features=self.features.categorical_indices,
                random_state=self.config.seed,
            )
            with self._thread_context():
                self.estimator.fit(x, y)
        else:
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError('Install ML dependencies: pip install -e ".[ml]"') from exc
            self.estimator = LGBMRegressor(
                objective="poisson", learning_rate=0.05,
                n_estimators=self.config.iterations, num_leaves=31,
                n_jobs=self.config.threads, random_state=self.config.seed,
                deterministic=True, force_col_wise=True, verbosity=-1,
            )
            self.estimator.fit(pd.DataFrame(x, columns=self.features.names), y,
                               categorical_feature=self.features.categorical_indices,
                               feature_name=self.features.names)
        return self

    def predict(self, horizon: int):
        if horizon != self.config.horizon:
            raise ValueError("Direct model predicts only the horizon used in training")
        if self.constant is not None:
            return np.full((len(self.features.panel.ids), horizon), self.constant)
        x = self.features.at_origin(len(self.features.panel.dates) - 1, horizon)
        if self.name == "lightgbm":
            x = pd.DataFrame(x, columns=self.features.names)
        with self._thread_context():
            predictions = self.estimator.predict(x)
        if not np.isfinite(predictions).all():
            raise ValueError("Model returned nonfinite predictions")
        return np.maximum(predictions, 0).reshape(len(self.features.panel.ids), horizon)


def make_model(name, config):
    if name in {"seasonal_naive", "mean28"}:
        return Baseline(name)
    if name in {"histgbm", "lightgbm"}:
        return DirectTree(name, config)
    raise ValueError(f"Unknown model: {name}")
