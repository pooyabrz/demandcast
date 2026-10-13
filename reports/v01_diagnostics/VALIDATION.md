# Validation receipt — diagnostic addon

Verified on 2026-10-12.

- The supplied 126,000 rows were checked for duplicate forecast keys, finite and nonnegative values, matching horizons/dates, identical actual sales across models, and complete paired model coverage.
- Independently recomputed fold-level MAE, WAPE, bias and observation counts match the earlier supplied metrics to numerical precision (maximum absolute deviation below 1e-12).
- The 42,000 distinct observations form 500 sampled M5 product–store series over three 28-day validation windows.
- Five notebook code cells executed sequentially with a shared Python namespace. The saved notebook passes nbformat schema validation, has three embedded plot outputs, and no cell errors.
- Plot PNGs were visually inspected for labels, scales, signs, units and clipping. Notebook text/HTML outputs were checked for bounded presentation and agreement with the generated CSV tables.
- The three focused ablation checks passed: a real LightGBM L1 component fit on synthetic fixture features, all-zero model behavior, and rejection of changed data/settings by the frozen-reference contract.

## Material limits

The notebook was executed through sequential Python cells, not an interactive Jupyter kernel. Its valid notebook file and outputs are shipped for execution in the user's notebook environment.

The original DemandCast installation and prepared historical m5.csv are not present in this workspace. Therefore the full installed-package experiment CLI and the L1 model's accuracy on real M5 data have not been executed here. The component tests inject explicit synthetic fixture features; they do not substitute for real training history. The experiment reuses the installed v0.1 feature/split code and checks that the original models reproduce before accepting the comparison.

Observed zero-sales and retrospective sparse-series groups are diagnostic slices, not future routing inputs. Their error contributions are arithmetic, not a causal explanation. Lower MAE does not establish an inventory policy; signed bias and error on positive-sales days remain decision checks.

## Source and definitions

Source predictions checksum is recorded in summary.json. reference_run.json and reference_metrics.csv preserve the user's validation reference. Holdout data is excluded from the analysis and experiment.

MAE is in unit-sales units. WAPE is a fraction and is undefined for a zero actual-sales denominator. net_extra_absolute_error is sum(abs(LightGBM-y) - abs(mean28-y)); positive values mean more LightGBM error. Rates and MAEs are recomputed from detail rows, not unweighted averages of subgroup rates.
