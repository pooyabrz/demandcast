# Reproduction contract

## Existing core checkout

The release addon is applied to the original DemandCast checkout; it is not a
replacement for its `pyproject.toml`, source package or data-preparation code.
`apply_release_files.py` retains the original README as
`CORE_SETUP_ORIGINAL.md` and any existing `ci.yml` as a text backup before
installing the final presentation and workflow. Other workflow files remain
available for deliberate review.

```bash
python scripts/apply_release_files.py
python -m pip install -r requirements-release.txt -e .
python scripts/project_preflight.py
python -m pytest -q tests --junitxml=reports/release/local-tests.xml
```

The observed benchmark versions are NumPy 2.4.2, pandas 3.0.0 and LightGBM
4.7.0. Dashboard entry points are Streamlit 1.65.0 and Plotly 6.9.0. The release
requirements pin these entries and pytest; they are not a full transitive
lockfile. The preflight report records the actual installed versions and code
hashes. Resolve any project-specific constraint conflict before benchmarking.

## Data and time splits

Use the original preparation code and fixed sample. Frozen ablations require
the exact prepared CSV hash recorded in the experiment log. Raw M5 files and
generated prediction tables are local inputs and are not part of the repository.
Small reference summaries remain tracked because the code and tests inspect them.

| Population | Cutoffs | Observations per model/method |
| --- | --- | ---: |
| E01/E02 point benchmark | 2016-01-31, 2016-02-28, 2016-03-27 | 42,000 |
| E02 interval benchmark | Same three cutoffs | 42,000 |
| E03 method comparison | 2016-02-28, 2016-03-27 | 28,000 |

The first E03 window supplies initial scores only. Later corrections use all
complete earlier windows whose final target is observed at the current cutoff.
Current and future target labels do not determine the current correction.
Rates and mean scores are pooled using observation-count weights; WAPE uses
total absolute error divided by total actual sales.

## Outputs and review

The ablation scripts save predictions, summaries, source hashes and run receipts
under `outputs`. The dashboard opens these when present and otherwise uses
bundled aggregate results. Series exploration requires the E02 file referenced
by E03's hash. Missing files produce an explicit empty state, and the interaction
tests create that state independently of real project outputs.

Preflight checks package interfaces, required files, installed dependencies,
reference reconciliation and the local prepared-data hash. CI mode skips M5
input checks. Git inspection reports already-tracked generated files because
adding ignore patterns does not remove files from an existing index.

## Verified scope

The available addons can be merged and their component/interaction checks run
locally; the combined suite passed all 32 addon checks. See
[validation evidence](VALIDATION.md) for the tested runtime and exact scope.
The original source checkout and `pyproject.toml` were not available
while authoring this release addon, so its original tests and package installation
are checked on the user's checkout and subsequently by CI. A YAML/template
review and local addon pass do not establish a remote workflow pass.

The dashboard's headless interaction tests are functional checks. Browser visual
review was blocked in the authoring environment; the user subsequently confirmed
the local dashboard and corrected tests worked. No unobserved screenshot or
independent M5 test performance is claimed.
