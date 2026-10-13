# Validation receipt — phase 3 interval addon

Verified on 2026-10-12.

## Existing L1 result review

- The uploaded L1 run is validation-only, uses the same prepared-file hash and configuration as the preceding experiment, reports reproduced controls, and did not evaluate holdout.
- Fold and sales-state summaries independently reconcile to the uploaded overall MAE, WAPE, bias and observation counts.
- All model summaries imply the same 57,003 observed unit sales across 42,000 validation observations.
- L1 reduces MAE by 9.6981% relative to the original Poisson model and by 9.4095% relative to mean28. It beats mean28 in each of three validation windows.
- Aggregate forecast-volume difference is -34.5031% for L1, -3.7013% for Poisson and -0.5508% for mean28.
- On positive-sales observations, L1 MAE is 14.8637% greater than Poisson; on zero-sales observations it is 56.2987% lower.

These are verified arithmetic comparisons on the sampled validation population. They are not an independent new test, causal proof, or an inventory-policy evaluation. Row-level L1 predictions were not uploaded in this turn; validation is limited to the supplied summaries and run record.

## New code verification

Six focused tests passed. They cover interval coherence and unchanged median estimates, raw versus adjusted coverage, width and pinball-loss calculations, invalid/missing model observations, real LightGBM quantile components on injected synthetic fixture features, changed frozen control scores, and changed frozen data/settings. Python sources compile successfully.

The tests execute the quantile-model components and reporting functions. The full CLI against the user's installed DemandCast and historical m5.csv has not run here because that installation and historical panel are unavailable. No real-M5 interval coverage or improvement is claimed in this package.

## Interpretation rules

Poisson and L1 outputs estimate different central functionals. Preserve both and assess each for its purpose. Do not treat a median as an unbiased estimate of aggregate expected sales.

The q10/q90 target is nominal 80% coverage. Inclusive bounds are used for observed count data. Empirical coverage can differ from 80%, and outcome-conditioned zero/positive-sales groups need not each have 80% coverage. Those groups are diagnostics, not future routing inputs.

Bounds are widened to contain q10, the frozen L1 median and q90. Raw coverage, crossing and adjustment rates remain visible. This deterministic coherence repair is not conformal calibration and has no distribution-free coverage guarantee. Holdout is excluded.

Sources: supplied overall_metrics.csv, metrics(2).csv, sales_state_metrics.csv and run(3).json, copied as frozen references in this directory. Technical documentation: https://lightgbm.readthedocs.io/en/stable/Parameters.html and https://otexts.com/fpp3/toolbox.html.
