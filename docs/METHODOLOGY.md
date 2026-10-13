# Forecasting and evaluation contract

## Target and population

The target is observed daily unit sales per M5 product–store series. It is not
revenue or unconstrained demand. Stockouts can censor sales, and inventory is
not modeled in v0.1. Series are selected deterministically from identifiers only;
no future sales influence selection. This sample is not asserted to represent
the entire M5 population.

## Time boundaries

The dataset has a shared, complete daily calendar. At each evaluation cutoff,
model fitting receives a panel ending at that cutoff. Validation uses earlier
nonoverlapping horizons; the final horizon is holdout and is never evaluated by
the default validation command. Every model uses the same series and dates.

Direct-model training examples have an origin, horizon, and target date. Origins
are sampled every seven days by default. History features stop at the origin;
labels stop at the end of the model's observed training prefix. Even for horizon
28, no observed sale within the forecast horizon becomes an input feature.

The direct model is pooled across series and horizons. It uses sales at origin,
sales 7/14/28 days before origin, trailing means/std/zero fraction at origin,
future calendar features, horizon, and categorical store/department/category/state.
Item ids are not used as arbitrary ordinal numeric features.

## Metrics

- MAE: mean absolute unit error, pooled over the stated rows.
- WAPE: total absolute error divided by total absolute observed sales.
  It is a fraction, not a value already multiplied by 100. A zero denominator
  gives an undefined value, written as an empty CSV field.
- Bias: mean(prediction - actual); a positive value means overprediction.
- Horizon and per-series tables pool the validation windows unless stated otherwise.

Compare MAE/WAPE across folds as well as aggregated rows. WAPE alone can hide
weaknesses on low-volume products. There is no universal guaranteed improvement
target. Official M5 WRMSSE needs its prescribed scale, weights, and aggregation;
it is deliberately deferred instead of approximated under the same name.

## Reproducibility and resource limits

Run manifests record config, library versions, cutoffs, dataset hashes, sampling
identifiers, and elapsed time. Equivalent environments should reproduce predictions;
byte-identical results across hardware and dependency versions are not promised.
The dependency ranges in pyproject are install constraints, not a resolved lock file.
After first successful local installation, record `python -m pip freeze` alongside
experiment results and choose a lockfile tool for the project.

M5 metadata is read before daily sales. Sales are scanned in bounded chunks;
only selected series are expanded to a long panel. Direct-model training can
multiply rows by the horizon, so history length and origin stride are explicit.

## Known limits of this milestone

No dashboard, API, model registry, intervals, SHAP, cold-start evaluation,
hierarchical reconciliation, deployment, or inventory policy is implemented yet.
Input series must share a complete calendar. Metadata is static. Prices are
excluded until availability at historical origins is documented. The final
holdout is separated by workflow, not hidden from a user who opens the raw file.

Freeze model settings before running holdout. Revising the model repeatedly in
response to holdout scores turns that holdout into another validation set.
