# Final product-cohort evaluation

Frozen recipe retrained on 50 fresh products across 10 stores.
Cutoff: 2016-04-24. Targets: 2016-04-25 to 2016-05-22.

| Model | MAE | WAPE | Aggregate volume bias |
| --- | ---: | ---: | ---: |
| seasonal_naive | 1.0550 | 103.46% | -5.97% |
| mean28 | 0.8978 | 88.04% | -8.12% |
| lightgbm | 0.9127 | 89.50% | -2.80% |
| lightgbm_l1 | 0.8214 | 80.55% | -47.32% |

Observations per model/interval: 14,000.
Envelope coverage: 89.85%; mean width: 2.4139; IS80: 3.7643.
Lower/upper miss rates: 0.00% / 10.15%.

L1 is median-oriented; Poisson is mean-oriented. Inspect expected volume separately from MAE.
Comparisons use the same new-cohort targets. Prior validation scores are not a paired before/after comparison.
The cohort is product-disjoint, while dates/store conditions are shared. This is not an independent dataset,
a statistical-significance result or an inventory service-level guarantee. This cohort is now inspected:
do not tune or resample and continue calling it untouched.
