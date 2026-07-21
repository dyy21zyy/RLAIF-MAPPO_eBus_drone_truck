# Experiment Readiness Report

Formal results not yet produced.

## Repository Commit

To be filled after commit: `1596ef831a77a253fd22c782b1037ca0cc37b595`.

## Environment Details

- Date: 2026-07-21 UTC.
- Working directory: `/workspace/RLAIF-MAPPO_eBus_drone_truck`.
- Python environment: local container used for smoke/integration gates.
- PyTorch runtime: checked by smoke commands; unavailable commands must be recorded as blockers rather than converted into validation claims.

## Implemented Architecture

The codebase implements an event-driven parcel-delivery simulator with release-time assignment, truck batch dispatch, physical electric-bus circulation, passenger dynamics, bus freight loading and charging decisions, station drone dispatch, explicit depleted/full/charging battery states, station-selected battery charging, asynchronous four-agent MAPPO surfaces, reward ledger accounting, multi-agent RLAIF schemas and wrappers, and formal experiment configuration files.

## Parameter Sources

Parameters come from committed YAML configs under `configs/` and `configs/paper/`. Formal paper-scale parameters are documented in `configs/paper/parameter_provenance.yaml`. Synthetic fallback fixtures are for smoke and CI only.

## Data Provenance

Committed `data/scenarios/*/*/instance.json` files include fixture provenance. Real transit files are not present and are not claimed. Preference data under `data/preference/` is not final AI-labeled training data.

## Commands Executed

This report is updated during the Phase 10 gate. Exit codes, warnings, counts, and blockers must be copied from the terminal, not inferred.

| Command | Exit code | Notes |
|---|---:|---|
| `python -m compileall -q .` | 0 | Passed. |
| `python -m pytest -q` | 0 | 265 passed, 3 skipped, 1 warning. |
| `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml` | 0 | Stage 1 smoke passed. |
| `python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback` | 0 | Stage 2 fallback pipeline passed; built 60 parcels and 421 passenger arrivals. |
| `python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback` | 0 | Stage 3 smoke passed; 88 decisions; invariants passed. |
| `python -m experiments.smoke_test_mappo_async` | 0 | PyTorch available; four-agent MAPPO smoke passed; checkpoint round-trip true. |
| `python -m experiments.smoke_test_reward_model` | 0 | Reward-model smoke passed with warning: only 10 usable preference labels, not final quality. |
| `python -m experiments.smoke_test_experiments` | 0 | Stage 8 smoke passed; learned checkpoint status skipped_missing_checkpoint. |
| `python -m experiments.train_policy_matrix --config configs/paper/benchmark.yaml --validate-only` | 0 | Policy matrix valid. |
| `python -m experiments.run_paper_ablation --config configs/paper/ablation.yaml` | 0 | Processed configured ablations with unavailable artifacts as explicit skips. |
| `python -m experiments.run_sensitivity --config configs/paper/sensitivity.yaml` | 0 | Processed 0 sensitivity episodes; no failed rows. |
| `git diff --check` | 0 | Passed. |
| `git status --short` | 0 | Showed expected modified/new files before commit. |

## Test Counts and Skips

`python -m pytest -q`: 265 passed, 3 skipped, 1 warning.

## Artifact Paths

- Scenario fixtures: `data/scenarios/`.
- Formal configs: `configs/paper/`.
- Readiness docs: `docs/EXPERIMENT_READINESS_REPORT.md`, `docs/paper_code_alignment/`.
- Expected future checkpoints: `results/checkpoints/` (not currently validated as formal artifacts).

## Checkpoint Availability

Formal trained checkpoints are missing. Smoke placeholder artifacts must not be treated as formal checkpoints.

## Preference-Data Availability

Final AI preference labels are missing. Smoke/replay fixtures do not establish preference-label quality.

## Real-Transit-Data Availability

Real transit files are missing from this repository checkout. Fallback data is synthetic and deterministic.

## Warnings and Blockers

- No final AI labels.
- No validated four reward-model checkpoints.
- No formal trained policy checkpoints.
- No large-scale experiment outputs.
- No hardware runtime validation for long formal training.
- Real transit files absent.

## Exact Next Steps Before Final Experiments

1. Install/verify the target PyTorch and experiment runtime on the intended hardware.
2. Add real transit files with provenance or document the exact external data retrieval process.
3. Collect final multi-agent preference labels for assignment, truck, bus, and station decisions.
4. Train and validate four reward models; save checkpoint provenance.
5. Train environment-reward and RLAIF MAPPO policies from disjoint training banks.
6. Run paired benchmark, ablation, and sensitivity suites on validation/test banks.
7. Archive configs, seeds, logs, checkpoints, metrics, and hardware details.
