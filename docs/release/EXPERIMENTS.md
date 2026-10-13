# Development experiment record

## Common controls

The prepared panel SHA-256 is
`9d37afc6331cd22cfb951e0d3f5ee090980626dcb7d3531faee693285eaf30c3`.
The panel has 500 series and 730 days. The seed is 42, horizon 28, training
origin history 365 days, origin stride 7, 200 tree iterations and 4 threads.
Validation cutoffs are 2016-01-31, 2016-02-28 and 2016-03-27.

All changes were evaluated against unchanged controls on the same targets.
Predictions are nonnegative. Feature construction and training originate from
the core package; the addon code uses its observed-history feature interface.

| Stage | Change | Evidence | Decision |
| --- | --- | --- | --- |
| E00 | Initial point baselines and Poisson | Three validation windows and the previously inspected v0.1 holdout | Retain point controls; document holdout exposure |
| E01 | LightGBM objective changes from Poisson to L1 | Same 42,000 validation observations and reproduced point controls | Keep L1 as a median-oriented estimate; retain Poisson as a mean-oriented estimate |
| E02 | Add q10 and q90 tails | Three windows; controls reproduced and no quantile/envelope adjustments observed | Keep the q10-q90 envelope for uncertainty review |
| E03 | Past-only horizon-band correction | Initial window calibrates; both methods evaluated on identical 28,000 later observations | All six corrections zero; neutral ablation, no promotion |
| Dashboard | Review filters and source-bound forecasts | Real aggregate references, fixture-based series interactions and user-confirmed local execution | Local validation-demo interface accepted |
| Additional final cohort | Frozen recipe on fresh products | Registration only; evaluation still pending | No new final-test performance claimed |

## Principal findings

L1 pooled MAE is 0.9561076434 versus 1.0554166662 for the 28-day mean and
1.0587906923 for Poisson. Its aggregate predicted-sales bias is -34.50%,
compared with -3.70% for Poisson. Median and mean outputs remain separate.

E02 interval coverage is 89.22% across all three windows. E03 coverage is
89.36% across only the two later windows. These different populations must not
be used to claim a calibration improvement. E03's two candidate methods have
identical coverage, width 3.0029619029 and interval score 4.4603462909.

The predicted-mean `2+` group has 4,865 observations, 85.98% coverage,
8.5828592512 units of width and 12.1330296020 interval score. Absolute
interval scores use different demand scales and do not establish relative
forecast skill by themselves. Retrospective zero/positive groups never route
a future forecast or calibration correction.

## Evidence locations

- `reports/v01_diagnostics`: original point reference and error diagnostics.
- `reports/l1_reference`: E01 point controls used by the quantile experiment.
- `reports/interval_reference`: supplied E02 interval reference tables.
- `reports/dashboard_reference/calibration`: supplied E03 aggregates and receipt.
- `reports/dashboard_reference/intervals`: E02 aggregates and receipt for the dashboard.
- `reports/dashboard_reference/selection.json`: validation-demo output policy.

The summaries and run receipts are original user-run evidence. Fixture tests
are code checks, not additional M5 outcomes. No remote Actions result or new
independent holdout score is inferred from local component tests.
