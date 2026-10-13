# Validation evidence and execution limits

## Supplied E02 evidence

The user supplied four aggregate CSVs and the E02 receipt. These are preserved
byte for byte alongside this note. The source data hash is unchanged from E01:
`9d37afc6331cd22cfb951e0d3f5ee090980626dcb7d3531faee693285eaf30c3`.

The receipt identifies 500 series, three validation windows, 28 forecast days,
LightGBM 4.7.0, reproduced point controls and no holdout evaluation in E02.
The uploaded point metrics also numerically match the frozen E01 summaries.

Independent arithmetic checks use Python's standard CSV and math modules, not
the calibration runner's metric helpers. Every weighted aggregate in the horizon
and actual-sales-state summaries agrees with the overall table. Counts total
42,000; covered counts are 37,473. Covered counts by band are 9,495, 9,299 and
18,679. Covered counts by actual state are 14,140 positive and 23,333 zero.
All four point summaries imply the same actual total, 57,003.

Coverage is 0.8922142857 and mean width is 2.9760198003 units. Crossing,
median-outside and bounds-adjustment rates are all zero. Raw and delivered
coverage agree, so envelope repair did not cause the observed overcoverage.
The signed interval-score identity gives an 80% interval score of 4.5127781396:
`10 * (pinball_q10 + pinball_q90)`. This equals the standard width-plus-tail-error
score because the raw ordered tails equal the delivered bounds in this run.

See `summary_checks.json` and rerun from the project root:

```bash
python analysis/verify_interval_summaries.py --reference-dir reports/interval_reference
```

## Calibration implementation checks

Ten focused unittest checks passed with the bundled Python runtime, NumPy and
pandas. They cover the two tail penalties and inclusive endpoints, finite-sample
order-statistic selection, insufficient calibration data, prevention of current
and future label leakage, incomplete-window exclusion, support and unchanged
central forecasts, zero-expansion behavior, bad prediction grids and envelopes,
changed input receipts, and a complete synthetic CSV input/output run. The
synthetic pipeline also rejects a deliberately changed frozen coverage summary.
The command-line entry point was exercised separately with the same synthetic
data and references. This is not an M5 model-training or M5 calibration result.

```bash
python -m unittest discover -s tests -p test_temporal_calibration.py -v
```

## Material limitations

- E02 row-level `interval_predictions.csv` was not attached. Its date alignment,
  per-series bounds, calibration corrections, tail miss rates and future-fold
  calibration performance cannot be checked from these aggregate tables alone.
- Full M5 calibration must run against that saved file in the user's project.
  The new runner will verify its grid and reproduce the supplied summaries
  before saving comparison metrics.
- First-window rows calibrate only. Both candidate methods are compared on the
  same 28,000 rows from the two remaining windows. The 42,000-row baseline is
  saved separately; do not compare its pooled score directly with a two-window
  candidate score.
- These already inspected validation windows support exploratory development.
  This replay is not a newly untouched test set. The original v0.1 holdout has
  also previously been inspected in this project; do not describe it as a new
  blind test in a release report.
- Scores are dependent across series/time and come from models fitted at
  different cutoffs. The split CQR exchangeability theorem does not establish
  coverage for this expanding-window replay. Horizon-band groups and predicted
  mean bins do not come with conditional coverage guarantees.
- Zero/positive actual sales are retrospective diagnostic groups. Their
  coverage need not individually equal the marginal 80% target. They never
  select a correction. Conservative count intervals can produce ties and
  higher-than-nominal inclusive coverage.
- The applied correction is expansion-only and may be zero. Coverage improvement
  does not by itself establish better intervals; assess width, interval score
  and both tail miss rates. No method is automatically promoted.

## Primary methodological sources

- Romano, Patterson and Candès (2019), *Conformalized Quantile Regression*:
  https://arxiv.org/abs/1905.03222 — signed scores, order-statistic correction,
  and the exchangeability requirement. The implementation adapts the score to
  an empirical past-only replay and clips negative correction values to zero.
- Gibbs and Candès (2021), *Adaptive Conformal Inference Under Distribution
  Shift*: https://arxiv.org/abs/2106.00170 — a separate online adaptive method;
  this addon does not implement its alpha update or inherit its guarantee.
