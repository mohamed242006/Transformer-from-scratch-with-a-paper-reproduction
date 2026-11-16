# ADR-016: Deliver code and data to Kaggle as private Kaggle Datasets with a content manifest

- **Status:** Accepted. Amended 2026-11-16: the owner restored the GitHub repository, so a `git clone` (KAGGLE.md §0) is now the primary delivery path; Kaggle Datasets remain the supported offline fallback.
- **Date:** 2026-11-15
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-010](ADR-010-compute-platform.md), [ADR-013](ADR-013-run-artifacts-and-resumption.md), [ADR-002](ADR-002-iwslt14-before-wmt14.md), [KAGGLE.md](../KAGGLE.md)

## Context
All training runs on Kaggle (ADR-010). Kaggle notebooks need the code, the raw data and the tokenizer. Constraints:
- The project owner removed version control from the project, so a commit SHA cannot identify the code version, and there is no repository to clone.
- Kaggle internet access is optional (and needs a verified account). Validation and training must work offline.
- Dataset licences (Multi30k/Flickr30k, IWSLT/TED) allow research use but not public redistribution.
- `/kaggle/input` is read-only, while `pip install -e` and run directories need a writable copy.

## Decision
1. **Two private Kaggle Datasets**, built by `scripts/package_for_kaggle.py`:
   - `aiayn-code`: the project tree with environments, data, caches, artifacts and `dist/` excluded (`aiayn.diagnostics.packaging`), plus `AIAYN_MANIFEST.json` (per-file SHA-256 and a **tree hash**).
   - `aiayn-<dataset>` (e.g. `aiayn-multi30k`): the pinned raw archives (re-verified against the config checksums), the trained tokenizer, `tokenizer_fingerprint.txt` and `SHA256SUMS`.
2. **The tree hash is the code version.** It is deterministic (independent of time, order and machine) and is recorded by `kaggle_validate.py` in every report's `environment.code_tree_sha256`. Training runs will record it the same way (ADR-013 metadata).
3. **The data bundle doubles as the offline mirror** for `prepare_data`/`kaggle_validate` (`--data-mirror`). Checksums are verified again on Kaggle.
4. Notebooks copy the code to `/kaggle/working/aiayn`, install with `pip install -e . --no-deps --no-build-isolation` (never touching Kaggle's torch), and fall back to `PYTHONPATH=src` if that fails.
5. Both datasets stay **private**.

## Consequences
**Positive:** works with internet off; identical code verifiably runs on the laptop and on Kaggle (same tree hash); Kaggle dataset versions give a rollback history. **Negative:** re-upload after every code change (one CLI command); no line-level diff history (Kaggle versions are snapshots). **Risk:** uploading a stale bundle → the report shows the tree hash; compare it with the local `package_for_kaggle.py` output before trusting results.

## Alternatives considered
| Alternative | Why rejected (for now) |
|---|---|
| `git clone` from GitHub inside the notebook | No repository (owner's decision); needs internet; a public repo would also expose unfinished work. Remains a good option later for the portfolio (see KAGGLE.md §6). |
| Paste code into notebook cells | Duplicates logic (ADR-010 §5), untestable, drifts |
| Publish the package to PyPI | Overkill; public; versioning burden |
| Ship processed text instead of raw archives | Loses the checksum chain to the original sources (ADR-002) |

## Compliance
`tests/unit/test_diagnostics.py`: bundle completeness (every module, config, script and test shipped; top-level `data/` excluded), deterministic tree hash. `tests/integration/test_kaggle_validate.py`: the offline-mirror + bundled-tokenizer path.
