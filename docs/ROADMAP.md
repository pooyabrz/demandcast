# Staged roadmap

| Milestone | Deliverable | Acceptance evidence |
|---|---|---|
| 0.1: foundation | Current importer, CLI, baselines, direct trees, and temporal validation | Contract tests and synthetic demo; real-data run still pending |
| 0.2: real benchmark | Focused M5 EDA, verified dataset manifest, frozen baseline/ML comparison | Per-fold, per-series, and per-horizon results with runtime/memory measurements |
| 0.3: uncertainty | Calibrated prediction intervals | Empirical coverage and interval width by horizon on unseen periods |
| 0.4: presentation | Streamlit demo, model explanations, reproducible run instructions | Screenshots, user-visible source/model dates, and a working local demo |
| 1.0: release | Batch forecast API if needed, Docker, environment lock, full README | Clean-machine setup, CI, final untouched holdout evaluation, and release notes |
| 1.1: distinctive extension | Choose hierarchy or inventory simulation | Coherence checks or policy-cost comparison with explicit assumptions |

Do not start all extensions at once. Prioritize the real-data benchmark before
adding a UI. Keep baseline comparisons even when adding more complex models.
Treat explanations as model behavior, not causal effects. Label simulation
inputs and outcomes separately from observed business results.
