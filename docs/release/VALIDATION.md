# Release addon validation

The four delivered addon source trees and this release addon were assembled in
a separate authoring QA directory. No substitute core package or project
configuration was created to make the checks pass.

## Completed checks

- Combined addon suite: **32 passed** with pytest 8.4.2. This includes 3 L1,
  6 interval, 10 temporal-calibration, 9 dashboard and 4 release-tool checks.
- Release-tool checks verify retained original documents, repeatable installation,
  exclusion of whole development products, selection unaffected by metadata row
  order or unused future outcomes, and rejection of duplicate/unknown identities.
- All four release Python sources parsed successfully.
- The CI YAML parsed successfully. Its push/PR/manual triggers, read-only
  repository permission, three environment entries and pytest step were checked.
- Preflight reconciled the real bundled E02/E03 reference receipts and summaries.
  It reported the absent core files/interfaces as errors and did not label this
  incomplete authoring checkout ready. CI mode performed no M5 input check.

Runtime: Linux, Python 3.12.14, NumPy 2.4.2, pandas 3.0.0, LightGBM 4.7.0,
scikit-learn 1.8.0, Streamlit 1.65.0 and Plotly 6.9.0. These component checks
reuse real aggregate references and synthetic test fixtures; they do not create
new M5 model results.

## Checks on the actual checkout

The original DemandCast source package, its pyproject.toml and M5 inputs were
not present in the authoring directory. Package installation, original core
tests, local prepared-input hash and registration against the full raw M5
catalogue must pass on the user's assembled checkout. The full local test count
may therefore exceed 32.

No remote GitHub Actions job has been observed. The matrix is configured for
Ubuntu Python 3.12/3.13 and Windows Python 3.13; the local pass is not a result
for every matrix entry. No fresh-product forecast has been scored yet.

