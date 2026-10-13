# Verification receipt — v0.1 foundation

Verified on 2026-10-12 in the preparation environment.

- All 14 unittest cases passed; no skipped tests in the final run.
- Both HistGBM and LightGBM were fitted and generated forecasts on synthetic data.
- The default 8-series, 240-day, 28-day-horizon demo completed with all four models.
- Duplicate/missing/invalid input checks, future-data mutation checks, shared
  cutoff boundaries, final-holdout separation, and metric semantics passed.
- The M5-shaped importer was tested with generated fixture CSVs. It selected
  the same ids after target values were changed and retained only the chosen
  series and requested dates.
- A wheel built successfully from the final Python source.
- GitHub Actions is configured for Python 3.11 and 3.12; it has not been run
  on a remote repository yet.

## Tested environment

| Component | Version |
|---|---|
| python | 3.12.14 |
| demandcast | 0.1.0 |
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.8.0 |
| lightgbm | 4.7.0 |
| threadpoolctl | 3.6.0 |

LightGBM's required Narwhals dependency was present (2.27.1) in the final checks.

## Scope of the evidence

These checks validate execution and key methodological boundaries. They do not
establish accuracy on real M5 data, official leaderboard performance, memory
usage on a particular laptop, or business value. No original M5 data was loaded
or redistributed in this package. Full real-data preparation, evaluation, and
resource measurement are the next milestone.

A physical-core detection warning occurred in the managed preparation runtime;
model execution completed successfully. A LightGBM feature-name warning found
in an earlier check was corrected before the final run.
