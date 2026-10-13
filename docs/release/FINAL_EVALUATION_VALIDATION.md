# Final-evaluation addon validation

The unchanged earlier addon source trees, release tools and this addon were
merged in a separate QA checkout. No replacement DemandCast core package,
pyproject.toml or prepared M5 data was created for the checks.

## Completed

The combined suite passed **43 tests**, comprising the previous 32 addon checks
and 11 new final-evaluation checks. The final sources were tested after the
checkout/import guards and repeated-score behavior were added.

The new checks cover:

- Chunked historical input is unchanged when an unused future column changes,
  and ignores a future value that cannot be parsed as a number.
- The registered calendar yields 702 history days and 28 targets; a shifted
  day/date mapping is rejected.
- The injected model factory receives only complete history through cutoff.
  A future observation in its training input is rejected.
- MAE, WAPE, signed bias, aggregate volume bias, pinball losses, inclusive
  interval misses and IS80 agree with independent hand calculations.
- Crossed tails retain the registered envelope policy and unchanged median.
- Zero-sales groups have undefined WAPE and relative volume bias.
- Missing/duplicate forecast or target keys, extra target columns in forecasts,
  incorrect horizons and invalid forecast values are rejected.
- Changed recipe, source files, additional forecasting code or dependency
  versions do not satisfy the registration binding.
- A changed frozen forecast or binding cannot pass the seal check.
- The same registration cannot start another forecast in a different output.
- An explicit verified-context scoring fixture creates result tables, summary,
  artifact hashes and a final receipt. Repeating its successful score retains
  byte-identical results and does not call the target reader again.

The scoring orchestration fixture injects its verified context explicitly; it
does not pretend that a toy core installation passed the real project preflight.
Existing component checks exercise real LightGBM L1 and quantile estimators.

Runtime: Linux/Python 3.12.14; NumPy 2.4.2, pandas 3.0.0, LightGBM 4.7.0,
scikit-learn 1.8.0, Streamlit 1.65.0, Plotly 6.9.0 and pytest 8.4.2.

The forecast CLI was also invoked on the incomplete QA checkout. It stopped
at the absent actual registration before creating a final forecast attempt.
This verifies the entry-point failure behavior, not a real training run or a
completed core-interface check for this new runner.

## Remaining on the real checkout

The original core source, actual registration, raw calendar and sales inputs
are available on the user's computer. The runner checks these actual inputs
and interfaces before fitting. Their full forecasting integration is not
claimed from fixtures. The complete local test count may exceed 43 because it
also includes the original core tests.

No fresh-product M5 score, Python 3.13/Windows pass for this new addon or remote
GitHub Actions result has been observed in the authoring environment. Existing
development references are not substituted for final-cohort results.
