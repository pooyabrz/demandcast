# Additional final product-cohort evaluation

Status: protocol and registration tool prepared; no new evaluation run.

## Question

Does the frozen development recipe preserve its point and interval behavior on
products whose outcomes did not participate in our development selection?
The original v0.1 holdout was already inspected. Reusing that same sample and
calling it a newly blind test would misstate its role.

## Cohort, selected before outcome review

1. Match the 500 development series to M5 catalogue identity columns.
2. Exclude every `item_id` represented in development, including its series
   in all other stores. Series-only exclusion is insufficient for this protocol.
3. Rank eligible product identifiers by SHA-256 of `20261013|item_id`.
4. Select the first 50 products and every catalogue store for each product.
5. Require zero item/series overlap, ten stores and 500 selected series.

Selection deserializes only `id`, `item_id` and `store_id` in the raw catalogue
and `unique_id` in the prepared development panel. Whole-file checksums bind
the file versions; target values are not used for ranking or acceptance.
Metadata row order and unused target columns cannot change the selected cohort.

```bash
python scripts/register_final_evaluation.py
```

The registration contains exact item/series IDs, exclusions, file hashes, code
hashes, installed versions and the frozen configuration. It is created with
exclusive file creation and cannot silently overwrite a prior registration.
Registration does not train a model, parse new target sales or produce a score.

## Frozen forecast recipe and dates

The prepared new-cohort panel uses 730 daily dates ending on 2016-05-22. Models
are trained only on the new cohort's history through 2016-04-24 and forecast
2016-04-25 to 2016-05-22. Historical sales of the new products are permitted;
this is a retrained recipe check, not zero-shot transfer of already-fitted models.

The development feature pipeline, seed 42, horizon 28, training-origin history
365, stride 7, iterations 200 and threads 4 remain fixed. Models are seasonal
naive, mean28, LightGBM Poisson, LightGBM L1, q10 and q90. The interval is the
envelope of q10, L1 median and q90. The neutral temporal expansion is not added.
Do not change the cohort or recipe after inspecting final outcomes.

## Metrics and interpretation

Primary outcomes are median MAE, mean-estimate WAPE/aggregate volume bias and
the 80% interval score. Also report inclusive coverage, width, both tail misses
and the observation counts. Compare reference models on this same new cohort.
Absolute old/new-cohort scores are not a paired improvement comparison.

Report unexpected degradation as a result. Any follow-up tuning creates a new
development iteration and cannot reuse this scored cohort as an untouched test.
Keep registration and final run receipts with the results; do not turn a numerical
improvement into a significance claim based on 14,000 dependent daily observations.

## Limits

Products are disjoint from model-selection outcomes, while dates and store-level
conditions are shared. This is not a new independent dataset, an iid guarantee
or an inventory-policy evaluation. A later external-data benchmark can answer
a different generalization question. No new-cohort performance is available yet.
