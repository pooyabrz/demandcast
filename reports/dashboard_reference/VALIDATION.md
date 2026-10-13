# Dashboard validation and execution scope

## Sources and numerical checks

The bundled E02 and E03 files are the actual user-run aggregate tables and
receipts, copied byte for byte. E03's E02 receipt hash matches the bundled E02
receipt. The prepared M5 source hash is unchanged. Arithmetic was independently
checked with Python's standard CSV and math modules before UI authoring.

Both methods have identical pooled, per-fold and predicted-mean-group metrics.
Weighted per-fold and group values reconcile to every pooled metric. Each method
has 28,000 observations: 25,020 covered, 169 below the lower bound and 2,811 above
the upper bound. All six signed-score quantiles and applied corrections are zero.
Correction ranks and chronological calibration dates match the stated protocol.
The dashboard's loader repeats the source, population and weighted checks.

The `2+` predicted-mean group contains 4,865 observations, coverage 0.8598150051,
width 8.5828592512 and interval score 12.1330296020. These absolute scales do not
by themselves establish weaker relative forecasting skill.

Point results use a separate three-window population with 42,000 observations
per model. The supplied MAE/WAPE/count values imply the common actual total of
57,003, which is used to recover aggregate volume bias. This derivation is
limited to the positive-denominator frozen M5 result.

## Functional checks

Nine unittest checks passed, including five data checks and four Streamlit
AppTest interaction checks. They cover weighted filters, separate populations,
changed receipt rejection, row-file hash/date validation, undefined WAPE on
zero-sales subsets, default and single-window views, an empty selection,
predicted-mean group filtering, point-model review, a missing row file, and
per-series horizon filtering with synthetic rows.

The tests use the actual bundled aggregate summaries where available. Only
row-level series interaction fixtures are synthetic; they remain inside temporary
test directories and are not presented as M5 forecasts or shipped as demo data.
AppTest runs the real Streamlit script and inspects its generated elements;
it does not render the charts in a browser JavaScript engine.

The local Streamlit server was launched and its health endpoint checked.
The endpoint returned HTTP 200 with body `ok`. All nine checks passed again
with NumPy 2.4.2 and pandas 3.0.0, matching the user's reported package versions.
Python sources compile. The pinned dashboard packages are Streamlit 1.65.0
and Plotly 6.9.0. QA uses Python 3.12; Python 3.13 on the user's computer is
not independently exercised here. Dependencies were installed in an isolated
QA directory, leaving the shared Python runtime unchanged.

## Visual verification limitation

A browser screenshot could not be completed. The available Playwright runtime
had no Chromium executable; the official browser download returned invalid
archive bytes. This is a browser environment blocker, not an AppTest failure.
No screenshot or browser-layout approval is claimed. Verify chart labels, spacing
and responsive layout after opening the local app. Browser screenshots were not
saved or included as deliverables.

## Model and release limitations

The addon is a local Python repository component, not a hosted Data Site.
The supplied inputs do not include E02 row-level interval predictions, so actual
M5 series charts cannot be checked during authoring. The user can open their
matching local E02 file; the dashboard verifies its E03 SHA-256 before parsing.

This is development validation, not a newly blind final evaluation. Calibration
was neutral in this replay and is not automatically promoted. No new model is
trained, no new holdout is evaluated and no inventory policy is validated.
The independent evaluation and final repository release remain later steps.
