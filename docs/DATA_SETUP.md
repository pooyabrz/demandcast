# Obtain and prepare M5 data

1. Use the [original M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   or the data references linked from the [organizer repository](https://github.com/Mcompetitions/M5-methods).
   Follow source-specific access requirements and terms.
2. Create `data/raw/m5/` and place original CSVs there.
3. The current importer requires `calendar.csv` and either
   `sales_train_evaluation.csv` (preferred) or `sales_train_validation.csv`.
   `sell_prices.csv` can be retained locally for a later milestone but is not
   used by version 0.1.
4. Run the preparation command in the README. It validates metadata and the
   calendar, chooses 500 ids by seeded hash order, keeps the last 730 days, and
   writes `data/processed/m5.csv` plus `m5.manifest.json`.
5. Run validation. Do not run holdout while selecting features and settings.

File naming matters: an official file called `sales_train_evaluation.csv`
contains observed training history; its name does not replace this project's
explicit local holdout protocol. No official private leaderboard replication
is claimed. Derived CSVs and raw data are ignored by Git. Never commit Kaggle
credentials. Inspect applicable data terms before sharing derivatives publicly.

For another daily dataset, use columns `unique_id`, `ds`, and `y`. Optional
static columns are `store_id`, `dept_id`, `cat_id`, and `state_id`. The current
version rejects missing days, duplicates, missing/negative/nonfinite sales, and
metadata that changes within a series rather than silently imputing them.
