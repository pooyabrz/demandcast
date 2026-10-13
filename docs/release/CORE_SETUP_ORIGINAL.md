# DemandCast

**Retail unit-sales forecasting with origin-safe multi-horizon evaluation.**

DemandCast predicts daily unit sales for product–store combinations and measures
performance against weekly seasonal-naive and trailing-mean baselines. The current
release is the executable **v0.1 foundation**, intended to grow into a documented
retail forecasting portfolio project.

## Current scope

- Deterministic M5 series sampling before expanding the data to long format.
- Strict daily-panel validation and source-file hashes.
- Shared rolling-origin validation windows; the last horizon is reserved for holdout.
- Weekly seasonal-naive and 28-day trailing-mean baselines.
- Pooled direct multi-horizon LightGBM / scikit-learn HistGBM models.
- MAE, WAPE, bias, and error breakdowns by horizon and series.
- Synthetic smoke test, leakage-boundary tests, and explicit run manifests.

**No real M5 performance claim is made yet.** The shipped demo is synthetic. The
dashboard, prediction intervals, official WRMSSE, and inventory simulation are
planned milestones, not implemented features. See [ROADMAP](docs/ROADMAP.md).

## Quick start

Run commands from this repository's root with Python 3.11 or newer. Start with
Python 3.11 or 3.12, which are the CI targets.

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install and verify:

```bash
python -m pip install -e ".[ml]"
python -m unittest discover -s tests -v
python -m demandcast demo --config configs/demo.yaml --output outputs/demo
```

The demo writes predictions, metric tables, and `run.json`. WAPE is a fraction:
`0.20` means 20%; it is undefined when total observed sales are zero.

For a baseline-only installation, install `-e .` and remove the ML model names
from a copy of the configuration. The default demo also uses HistGBM and needs
the ML dependencies.

## Real M5 workflow

Download the original competition CSVs yourself according to the source terms.
See [DATA_SETUP](docs/DATA_SETUP.md). The starter does not redistribute M5 data.

```bash
python -m demandcast prepare-m5 --input-dir data/raw/m5 --config configs/m5.yaml
python -m demandcast backtest --data data/processed/m5.csv --config configs/m5.yaml --output outputs/m5-validation
```

Start with 500 series, 730 observed days, 4 threads, and weekly sampled training
origins. Measure runtime and memory before increasing the sample; these defaults
are a starting configuration, not a memory guarantee.

Only after model choices and settings are frozen, evaluate the final holdout:

```bash
python -m demandcast backtest --data data/processed/m5.csv --config configs/m5.yaml --stage holdout --output outputs/m5-holdout
```

After evaluation, refit on the available observed history to forecast new dates:

```bash
python -m demandcast forecast --data data/processed/m5.csv --config configs/m5.yaml --model lightgbm --output outputs/m5-future
```

These sample-level metrics are **not official M5 WRMSSE** or leaderboard scores.

## Why the time boundary matters

For a cutoff date `t`, all 28 forecast horizons use sales observed through `t`.
Future calendar dates are known. Observed sales after `t` are evaluation labels
and cannot become features. A pooled direct model includes the horizon as an
input; it does not consume actual intermediate future sales.

HistGBM's random internal early-stopping split is explicitly disabled. Tree
iteration counts are fixed and compared using the outer temporal validation.
Version 0.1 excludes dynamic prices because availability of future prices at
the historical forecast origin has not been established.

Read the full [methodology](docs/METHODOLOGY.md) before interpreting results.

## Repository map

| Path | Purpose |
|---|---|
| `src/demandcast/` | Import, features, models, evaluation, and CLI |
| `configs/` | Demo and M5 experiment settings |
| `tests/` | Leakage, data contracts, metric semantics, and integration |
| `docs/` | Dataset setup, methodology, roadmap, and Persian next steps |
| `notebooks/` | Reserved for focused real-data exploration |
| `data/`, `outputs/` | Local generated content; ignored by Git |

## Sources

- [M5 organizer repository: data, setup, and benchmarks](https://github.com/Mcompetitions/M5-methods)
- [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [LightGBM regressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

## Project status and ownership

This is a starter, with no GitHub repository or remote publication created by
this package. Add your author name, select a code license, and replace this status
section with your verified real-data results as the project progresses. No code
license is chosen on your behalf. Dataset terms remain separate from code licensing.
