# GitHub publication snapshot

Use `scripts/build_github_export.py` from the original assembled checkout after
the registered final cohort has been scored. It builds a new sibling directory;
it does not copy Git history, initialize Git, configure identity or push.

The actual original core checkout is needed. Local preflight must pass, the
forecast code/configuration must still match the completed run, and all original
forecast/result checksums must reconcile. Public references contain aggregate
tables, immutable registration and original receipts. Large per-observation
tables and raw/prepared M5 data stay in the original checkout. The receipt's
hashes for those omitted files are retained and explained by the reference README.
Integrity checks do not establish an independent replication of the experiment.

The exported README preserves development evidence, replaces the pending final
section with actual new-cohort values, and links a chart rendered from those
same point summaries. Median and mean estimates remain distinct. Old/new-cohort
absolute scores do not form a paired improvement. The exporter adds the plot
dependency to the copied CI workflow and its template without changing the
original checkout's workflow or forecast implementations.

Public-file selection includes core/source trees, tests, configs, documentation,
required package/setup files and the known reference directories. Local environments,
Git metadata, runtime outputs, raw/prepared inputs, environment files and binary
model/ZIP artifacts are omitted. The single-file budget is 50 MiB for this snapshot.
Existing license documents are retained; no replacement license is inferred.

A generated `.gitattributes` unsets `text` for all files. Experiment receipts
bind original bytes, so line-ending conversion must not silently alter hashed
reference JSON/CSV or frozen Python sources. A test commits and locally clones
a real E02 receipt with `core.autocrlf=true` and verifies the original SHA-256.
No global Git settings are changed by that check.

The export manifest records file hashes and can be checked with
`python scripts/build_github_export.py --verify-export` before initial publication.
It captures this snapshot; deliberate future changes need an updated provenance
record rather than being represented as byte-identical to the initial export.

A fresh repository can be created and pushed with Git or GitHub CLI after local
review. The tool never overwrites an existing destination, modifies an existing
remote, force-pushes, creates a tag or claims a remote workflow result. When
publishing to a repository with existing history, use that exact repository's
normal branch/PR workflow rather than transplanting this snapshot over its history.

Primary implementation references:

- [Import locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
- [GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create).
- [Git attributes and line-ending conversion](https://git-scm.com/docs/gitattributes).
- [GitHub file-size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
