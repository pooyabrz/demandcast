"""Small, explicit configuration; unknown keys fail instead of being ignored."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

MODEL_NAMES = {"seasonal_naive", "mean28", "histgbm", "lightgbm"}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    horizon: int = 28
    n_windows: int = 3
    n_series: int = 500
    history_days: int = 730
    training_origin_days: int = 365
    origin_stride: int = 7
    iterations: int = 200
    threads: int = 4
    models: tuple[str, ...] = ("seasonal_naive", "mean28")

    def __post_init__(self):
        for field in fields(self):
            if field.name == "models":
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value < (0 if field.name == "seed" else 1):
                raise ValueError(f"{field.name} must be a valid nonnegative/positive integer")
        if isinstance(self.models, str) or not self.models:
            raise ValueError("models must be a nonempty list of model names")
        object.__setattr__(self, "models", tuple(self.models))
        if len(set(self.models)) != len(self.models) or set(self.models) - MODEL_NAMES:
            raise ValueError(f"models must be unique names from {sorted(MODEL_NAMES)}")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(payload) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return Config(**payload)
