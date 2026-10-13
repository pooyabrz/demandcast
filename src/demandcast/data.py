"""Balanced daily panels, reproducible synthetic data, and bounded M5 import."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATIC_COLUMNS = ("store_id", "dept_id", "cat_id", "state_id")
M5_METADATA = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Panel:
    ids: np.ndarray
    dates: pd.DatetimeIndex
    values: np.ndarray  # series x dates, sorted lexically by id
    metadata: pd.DataFrame

    def prefix(self, end_index: int) -> "Panel":
        if not 0 <= end_index < len(self.dates):
            raise ValueError("Prefix boundary is outside the observed panel")
        return Panel(self.ids, self.dates[: end_index + 1],
                     self.values[:, : end_index + 1], self.metadata)


def to_panel(frame: pd.DataFrame) -> Panel:
    required = {"unique_id", "ds", "y"}
    if not required <= set(frame):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame))}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    data = frame.copy()
    if data[list(required)].isna().any().any():
        raise ValueError("Identifiers, dates, and observations cannot be missing")
    data["unique_id"] = data["unique_id"].astype(str)
    data["ds"] = pd.to_datetime(data["ds"], errors="raise")
    if data["ds"].dt.tz is not None or not data["ds"].eq(data["ds"].dt.normalize()).all():
        raise ValueError("Dates must be timezone-naive daily timestamps at midnight")
    if data.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate series/date observations")
    data["y"] = pd.to_numeric(data["y"], errors="raise")
    if not np.isfinite(data["y"]).all() or (data["y"] < 0).any():
        raise ValueError("Unit sales must be finite and nonnegative")
    dates = pd.DatetimeIndex(sorted(data["ds"].unique()))
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
        raise ValueError("Calendar has missing days; investigate instead of filling blindly")
    counts = data.groupby("unique_id")["ds"].nunique()
    if not counts.eq(len(dates)).all():
        raise ValueError("All series must share the complete calendar in version 0.1")
    static = [column for column in STATIC_COLUMNS if column in data]
    if static:
        if data[static].isna().any().any():
            raise ValueError("Static metadata cannot be missing")
        if (data.groupby("unique_id")[static].nunique() > 1).any().any():
            raise ValueError("Static metadata changed within a series")
    wide = data.pivot(index="unique_id", columns="ds", values="y").sort_index()
    metadata = data.drop_duplicates("unique_id").set_index("unique_id")[static]
    metadata = metadata.reindex(wide.index).astype(str)
    return Panel(wide.index.to_numpy(), dates,
                 wide.reindex(columns=dates).to_numpy(dtype=np.float32), metadata)


def synthetic_sales(n_series: int = 8, days: int = 240, seed: int = 42) -> pd.DataFrame:
    if n_series < 1 or days < 1:
        raise ValueError("Synthetic series and days must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    records = []
    for index in range(n_series):
        level = 5 + 3 * index
        weekday = 1 + 0.30 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
        trend = 1 + np.linspace(0, 0.15, days)
        values = rng.poisson(level * weekday * trend)
        if index % 3 == 0:
            values[rng.random(days) < 0.4] = 0
        records.append(pd.DataFrame({
            "unique_id": f"SYNTHETIC_{index:03d}", "ds": dates, "y": values,
            "store_id": f"STORE_{index % 2}", "dept_id": f"DEPT_{index % 3}",
            "cat_id": f"CAT_{index % 2}", "state_id": "SYNTHETIC_REGION",
        }))
    return pd.concat(records, ignore_index=True)


def prepare_m5(input_dir: str | Path, output: str | Path, config) -> dict:
    root = Path(input_dir)
    sales_path = root / "sales_train_evaluation.csv"
    if not sales_path.is_file():
        sales_path = root / "sales_train_validation.csv"
    calendar_path = root / "calendar.csv"
    if not sales_path.is_file() or not calendar_path.is_file():
        raise FileNotFoundError("Need calendar.csv and an original M5 sales_train_*.csv")
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    if not set(M5_METADATA) <= set(header):
        raise ValueError("M5 sales file is missing metadata columns")
    day_columns = sorted([c for c in header if c.startswith("d_") and c[2:].isdigit()],
                         key=lambda c: int(c[2:]))
    if len(day_columns) < config.history_days:
        raise ValueError("Requested history exceeds the available M5 history")
    day_columns = day_columns[-config.history_days:]
    metadata = pd.read_csv(sales_path, usecols=M5_METADATA, dtype=str)
    if metadata["id"].duplicated().any():
        raise ValueError("M5 series ids must be unique")
    if config.n_series > len(metadata):
        raise ValueError("Requested more series than the dataset contains")
    # Sampling uses identifiers only, never target values from any date.
    ranked = metadata["id"].map(lambda uid: hashlib.sha256(
        f"{config.seed}:{uid}".encode()).hexdigest())
    chosen = set(metadata.loc[ranked.sort_values(kind="stable").index[:config.n_series], "id"])
    selected_chunks = []
    for chunk in pd.read_csv(sales_path, usecols=M5_METADATA + day_columns,
                             dtype={c: "float32" for c in day_columns}, chunksize=256):
        subset = chunk.loc[chunk["id"].isin(chosen)]
        if not subset.empty:
            selected_chunks.append(subset)
    selected = pd.concat(selected_chunks, ignore_index=True)
    long = selected.melt(id_vars=M5_METADATA, value_vars=day_columns,
                         var_name="d", value_name="y")
    calendar = pd.read_csv(calendar_path, usecols=["d", "date"])
    if calendar["d"].duplicated().any():
        raise ValueError("Calendar day keys must be unique")
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    long = long.rename(columns={"id": "unique_id", "date": "ds"})
    long = long[["unique_id", "ds", "y", *STATIC_COLUMNS]].sort_values(["unique_id", "ds"])
    panel = to_panel(long)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(destination, index=False)
    manifest = {
        "dataset_kind": "real_m5_subset", "seed": config.seed,
        "sampling": "identifier SHA256 ordering; independent of target values",
        "n_series": len(panel.ids), "history_days": len(panel.dates),
        "start_date": str(panel.dates[0].date()), "end_date": str(panel.dates[-1].date()),
        "selected_ids": sorted(chosen), "source_sales_file": sales_path.name,
        "source_sales_sha256": sha256_file(sales_path),
        "source_calendar_sha256": sha256_file(calendar_path),
        "prepared_sha256": sha256_file(destination),
        "features_excluded": ["future prices", "future observed sales", "inventory"],
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
