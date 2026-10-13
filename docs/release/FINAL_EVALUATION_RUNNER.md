# Registered final-cohort runner

This runner implements the earlier protocol rather than defining a second
selection rule. It requires the actual DemandCast checkout and previous addons.
The registration remains unchanged, including its `registered_not_evaluated`
status; a separate attempt record and final receipt record the run's progress.

## Forecast phase

```bash
python experiments/run_final_evaluation.py --phase forecast
```

The runner checks the passing local preflight, original prepared-panel hash,
raw sales-file hash, exact registered identities, source hashes, installed
versions and frozen recipe. The execution interpreter/platform are also recorded
and must match between the two phases. All registered Python sources must remain unchanged.
The only new forecasting-source file allowed is this runner. Its hash and those
of the two release helpers are additionally bound when forecasting begins.

The sampler is replayed using identity columns only, and must reproduce the
exact registered cohort. An existing output or an existing attempt for this
registration prevents a new forecast run, including a run to a different output.
The script must live in the checkout whose code it verifies.

Calendar numbering is checked against M5's `d_1 = 2011-01-29`. The 730-day
interval ending 2016-05-22 is divided into 702 observed days through 2016-04-24
and 28 targets. Raw wide input is read in chunks of 256 rows with `usecols`.
Only identity metadata and historical sales columns are deserialized during
forecasting. Checksums read whole-file bytes; they do not supply target values
to selection, features, training or prediction.

The core `to_panel`, `OriginFeatures` and `make_model` interfaces remain in use.
L1 and quantile classes are imported from the unchanged objective addons.
Training receives a history-only panel. Forecasts are checked for finite,
nonnegative values and the complete six-model/500-series/28-horizon grid.
`forecasts.csv` contains 84,000 rows and has no `y` column.

The phase freezes forecast and receipt hashes before any target sales columns
are loaded. No final forecasts are fitted from a full panel whose future labels
are merely hidden behind a filtering convention.

## Score phase

```bash
python experiments/run_final_evaluation.py --phase score
```

Inputs, code, versions, attempt record, forecast receipt and forecast bytes must
still match. Only then are the registered target columns loaded and joined by
series/date to the frozen forecast grid. No model fitting occurs in this phase.
The final receipt binds all emitted result files. A repeated successful score
verifies and retains the existing result without loading targets again.

Scoring uses 14,000 observations per model/interval. Point MAE and WAPE use
unweighted daily observations with WAPE equal to total absolute error divided
by total actual sales. Aggregate volume bias is `(sum(prediction)-sum(y))/sum(y)`.
WAPE and relative volume bias are undefined when the denominator is zero.
Tail forecasts are assessed with pinball losses, without the optional factor 2.

The interval policy remains `min(q10, L1, q90)` to `max(q10, L1, q90)`.
Coverage is inclusive. IS80 is interval width plus ten times each out-of-bounds
distance. Raw crossings and envelope adjustments remain visible. The temporal
expansion is omitted because its development ablation was neutral, not because
the final targets were used to choose it.

Results include pooled summaries and horizon/store diagnostics, as well as
predicted-mean bins and retrospective zero/positive sales groups. These
retrospective groups never choose the deployed forecast or route its bounds.
Control comparisons use the same new-cohort targets. Positive relative
improvement means lower error; negative means worse error. Comparisons against
old-cohort absolute values are not treated as paired improvements.

## Artifacts and limits

Outputs live under `outputs/final-cohort`, already covered by the release
ignore rules. The small attempt record in `reports/final_evaluation` is an
audit record; keep it until the evaluation has been reviewed. Preserve the
registration and attempt if a phase fails rather than deleting them and silently
starting a fresh test. Scoring exposure is recorded before opening target sales;
on a failed read, full deserialization may remain unknown in the receipt.

The earlier final window on development products had been inspected. This
product-disjoint cohort is an additional frozen-recipe check with shared dates
and store shocks, not a new independent dataset or zero-shot transfer. After
scoring, its outcomes are inspected. Further tuning creates a development
iteration and requires a separately defined evaluation plan.

No actual new-cohort score or remote CI result is included in this addon.
The original core and raw inputs are executed on the user's checkout.

Primary method/data references:

- [M5 organizer data and methods](https://github.com/Mcompetitions/M5-methods).
- [M5 original data description](https://www.kaggle.com/competitions/m5-forecasting-uncertainty/data).
- [Quantile and Winkler scores](https://otexts.com/fpp3/toolbox.html#evaluating-distributional-forecast-accuracy).
