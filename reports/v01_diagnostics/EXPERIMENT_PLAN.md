# Experiment E01: align the training objective with absolute error

Status: planned for the local historical M5 panel; no M5 L1 result is claimed.

Question: does changing LightGBM's Poisson objective to regression_l1 reduce validation MAE against both the original LightGBM and mean28?

Evidence: 56.15% of validation observations are zero. On those observations, the original model adds 929.2 absolute-error units versus mean28; on positive sales it saves 787.5 units, leaving a +141.7 deficit. This supports an objective-alignment hypothesis without proving it.

Only objective changes. Keep the prepared-file hash, 500 series, 730 observed days, seed 42, horizon 28, three validation windows, training-origin window 365, stride 7, 200 iterations and four threads fixed. Reuse v0.1 features and cutoff logic.

Compare pooled MAE/WAPE, all three folds, horizon bands, paired per-series differences, zero/positive sales, and signed bias. Treat a pooled reduction against both controls plus wins on at least two folds as evidence worth continuing; do not label it statistically significant. Inspect worsening positive-sales error or negative bias before deciding suitability. No business threshold for acceptable bias has been established.

The script validates frozen input/configuration and reproduces the old controls within its recorded numerical tolerance. If reproduction fails, do not attribute a score change to objective alone. It never evaluates holdout.

Technical reference: https://lightgbm.readthedocs.io/en/stable/Parameters.html documents regression_l1 as L1/mean absolute error. Forecasting: Principles and Practice (https://otexts.com/fpp3/toolbox.html) explains that MAE-optimal point forecasts target the median; this can differ from mean-oriented sales aggregation.
