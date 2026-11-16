# ADR-018: Experiment composition, compatibility checks and resume-safe config hashing

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** Lead engineer
- **Related:** [ADR-007](ADR-007-configuration-system.md), [ADR-013](ADR-013-run-artifacts-and-resumption.md), spec §9

## Context
ADR-007 deferred composition across config groups to "the first schema that needs it". The trainer needs data, model and training settings together, and three inconsistencies can only be caught across groups: positional-encoding coverage (`max_positions` must cover the longest sentence + 50 for inference, paper §6.1), vocabulary agreement, and whether a resumed run is really the same experiment.

## Decision
1. **Experiment file** (`configs/experiment/*.yaml`):
   ```yaml
   name: multi30k_tiny
   data: multi30k              # -> configs/data/multi30k.yaml (or a repo-relative path)
   model: transformer_tiny     # -> configs/model/transformer_tiny.yaml
   train: multi30k_t4          # -> configs/train/multi30k_t4.yaml
   overrides:                  # optional, group-prefixed
     - data.batching.max_tokens=4096
   ```
   Command-line overrides use the same `group.key=value` form and are applied after the file's overrides.
2. **The resolved experiment** (all three groups, after overrides) is written to `runs/<id>/config.yaml` and is the only source for a resume.
3. **Compatibility checks** run before training: `max_positions ≥ longest data sequence + 50`, `max_tokens ≥` the longest sequence, AMP mode supported by the device, and the tokenizer vocabulary equal to the configured size.
4. **Config hash** = SHA-256 of the resolved experiment with the **resume-mutable keys removed**: `train.{max_updates, max_epochs, time_budget_minutes, log_every, loggers, device}`. A resume may change only those keys (e.g. extend training, change the session budget). Anything else changes the hash, and the resume is refused.

## Consequences
**Positive:** one file per reported result; mistakes across groups fail before any GPU time; extending a run is a legitimate override. **Negative:** another small loader to maintain. **Risk:** someone needs to change a non-mutable key mid-run → that is a different experiment and must start a new run (by design).

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Hydra defaults lists | Rejected in ADR-007 |
| One monolithic config per run | Duplicates data/model settings across dozens of files |
| Hash everything | Could not extend `max_updates` or change the session budget on resume |

## Compliance
`tests/unit/test_experiment.py`: every experiment file loads and passes compatibility; mismatches raise; the hash ignores exactly the mutable keys.
