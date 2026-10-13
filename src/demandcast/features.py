"""Direct pooled features: every horizon sees history through its origin only."""

import numpy as np
import pandas as pd

from demandcast.data import Panel

HISTORY_FEATURES = ["sales_at_origin", "sales_origin_minus7", "sales_origin_minus14",
                    "sales_origin_minus28", "mean7_at_origin", "mean28_at_origin",
                    "std28_at_origin", "zero_fraction28_at_origin"]
CALENDAR_FEATURES = ["horizon", "weekday_sin", "weekday_cos", "month_sin",
                     "month_cos", "day_of_month"]


class OriginFeatures:
    def __init__(self, panel: Panel):
        self.panel = panel
        self.static_names = list(panel.metadata.columns)
        self.mappings = {
            col: {value: index for index, value in enumerate(sorted(panel.metadata[col].unique()))}
            for col in self.static_names
        }
        self.names = HISTORY_FEATURES + CALENDAR_FEATURES + self.static_names
        self.categorical_indices = list(range(len(HISTORY_FEATURES) + len(CALENDAR_FEATURES),
                                               len(self.names)))
        self.static = np.column_stack([
            panel.metadata[col].map(self.mappings[col]).to_numpy()
            for col in self.static_names
        ]) if self.static_names else np.empty((len(panel.ids), 0))

    def at_origin(self, origin: int, horizon: int) -> np.ndarray:
        if not 28 <= origin < len(self.panel.dates) or horizon < 1:
            raise ValueError("Features need 29 observed days and a positive horizon")
        y = self.panel.values
        past28 = y[:, origin - 27:origin + 1]
        historical = np.column_stack([
            y[:, origin], y[:, origin - 7], y[:, origin - 14], y[:, origin - 28],
            y[:, origin - 6:origin + 1].mean(axis=1), past28.mean(axis=1),
            past28.std(axis=1), (past28 == 0).mean(axis=1),
        ])
        dates = pd.date_range(self.panel.dates[origin] + pd.Timedelta(days=1),
                              periods=horizon, freq="D")
        weekday = dates.dayofweek.to_numpy()
        month = dates.month.to_numpy()
        calendar = np.column_stack([
            np.arange(1, horizon + 1), np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7), np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12), dates.day.to_numpy(),
        ])
        return np.column_stack([
            np.repeat(historical, horizon, axis=0),
            np.tile(calendar, (len(self.panel.ids), 1)),
            np.repeat(self.static, horizon, axis=0),
        ]).astype(np.float32)

    def training_matrix(self, horizon: int, origin_days: int, stride: int):
        if min(horizon, origin_days, stride) < 1:
            raise ValueError("Training horizon, origin days, and stride must be positive")
        last_origin = len(self.panel.dates) - horizon - 1
        first_origin = max(28, last_origin - origin_days + 1)
        if first_origin > last_origin:
            raise ValueError("Need at least 29 observed feature days plus the training horizon")
        origins = np.arange(last_origin, first_origin - 1, -stride)[::-1]
        blocks = [self.at_origin(int(origin), horizon) for origin in origins]
        labels = [self.panel.values[:, origin + 1:origin + horizon + 1].reshape(-1)
                  for origin in origins]
        # The latest label ends at the final observed training date, never beyond it.
        return np.concatenate(blocks), np.concatenate(labels), origins
