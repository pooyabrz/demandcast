# DemandCast v0.2 development model card

## Task and intended use

Forecast daily observed unit sales for a sampled M5 panel over a 28-day horizon.
The current model and dashboard support a reproducible forecasting portfolio
project, validation diagnostics and comparison of point and interval forecasts.
Inventory service levels, lost demand and deployment outcomes are not measured.

## Data and protocol

The prepared panel contains 500 deterministically sampled series and 730 days.
Its SHA-256 is
`9d37afc6331cd22cfb951e0d3f5ee090980626dcb7d3531faee693285eaf30c3`.
Validation cutoffs are 2016-01-31, 2016-02-28 and 2016-03-27, with 28 targets
per series and window. Sampling and training details remain in the original
DemandCast pipeline; this dashboard does not refit any model.

E03 uses the first window for initial calibration scores only and compares both
methods on the two later windows. Complete earlier out-of-sample forecast
windows supply scores, separately within fixed horizon bands 1-7, 8-14 and
15-28. The correction is expansion-only. Current-window labels do not select
its correction. The final-window correction can use the previously completed
evaluation window, consistent with the stated replay protocol.

## Candidate models and outputs

- Seasonal naive and 28-day mean: reference point forecasts.
- LightGBM Poisson: mean-oriented point estimate.
- LightGBM L1: median-oriented point estimate.
- LightGBM quantile objectives with alpha 0.1 and 0.9: tail estimates. The
  delivered bounds are the envelope of these tails and the unchanged L1 median.

The mean and median are different forecasting targets. The demo presents both
and does not route a future forecast using its subsequently observed zero or
positive sales state. Pointwise daily interval bounds are not summed to claim
a calibrated interval for total sales.

## Development validation results

Point results cover 42,000 observations across three windows:

| Model | MAE | WAPE | Aggregate predicted-sales bias |
| --- | ---: | ---: | ---: |
| LightGBM L1 | 0.9561 | 70.45% | -34.50% |
| LightGBM Poisson | 1.0588 | 78.01% | -3.70% |
| 28-day mean | 1.0554 | 77.76% | -0.55% |
| Seasonal naive | 1.2530 | 92.32% | -2.88% |

Both interval methods cover the same 28,000 observations across two windows:

| Interval metric | q10-q90 envelope | Temporal correction |
| --- | ---: | ---: |
| Inclusive coverage | 89.36% | 89.36% |
| Mean width in sales units | 3.0030 | 3.0030 |
| 80% interval score | 4.4603 | 4.4603 |
| Lower miss rate | 0.60% | 0.60% |
| Upper miss rate | 10.04% | 10.04% |
| Mean applied expansion | 0 | 0 |

All six signed score quantiles and applied corrections were zero. The envelope
is therefore retained as the demo default; the temporal correction is recorded
as a neutral ablation. No statistically significant benefit is claimed.
For the predicted-mean `2+` group, coverage is 85.98% on 4,865 observations,
mean width is 8.583 units and interval score is 12.133. These absolute scores
have different demand scales and do not alone establish worse relative skill.

## Metrics

MAE is mean absolute error. WAPE is total absolute error divided by total actual
sales. Aggregate volume bias is `(sum(prediction)-sum(actual))/sum(actual)`.
Inclusive interval coverage counts `lower <= y <= upper`. The 80% interval
score is `width + 10*max(lower-y, 0) + 10*max(y-upper, 0)`; lower is better.
Subgroup and window summaries are pooled with observation-count weights.

## Limitations and next release gate

This is development validation on windows already inspected during model
selection. E03 did not evaluate the holdout. The original v0.1 holdout was
previously inspected and must not be described as a newly blind test. A future
release needs a preregistered, independent evaluation design and a clear record
of all selection decisions.

Temporal and cross-series dependence and changing fitted base models mean
that this replay does not inherit split CQR's exchangeability-based guarantee.
Nominal 80% describes a target; actual count-valued coverage can be conservative.
Groups based on future zero/positive outcomes are retrospective diagnostics and
need not individually attain 80% coverage. The nominal level is not an inventory
service level. This development result is not an official M5 leaderboard score
and does not report full hierarchical WRMSSE.

Full per-series review needs the E02 row-level interval file. Only aggregate
E03 CSVs and receipts were supplied while preparing this addon; no new M5 model
training or independent test was performed here.

## Evidence and reproduction

The original user-run receipts and aggregate tables are preserved under
`reports/dashboard_reference`. The dashboard checks source hashes, count and
weighted-metric reconciliation before displaying results. The source review
shows the files and definitions supporting the current view.

Primary references: [CQR](https://arxiv.org/abs/1905.03222),
[forecast evaluation](https://otexts.com/fpp3/toolbox.html),
[Streamlit](https://docs.streamlit.io/) and [Plotly](https://plotly.com/python/).
