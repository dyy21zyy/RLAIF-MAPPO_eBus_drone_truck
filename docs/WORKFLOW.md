# Development Workflow

## Stage gates

Each stage must pass its own offline smoke test before downstream work begins.
A stage gate is mandatory at the end of every stage. If it finds a blocker, fix
the current stage immediately, add regression coverage, and rerun the complete
gate. Do not begin the next stage until the gate passes. Automatic remediation
may run for at most three rounds before the remaining blockers are escalated for
manual intervention.

1. **Stage 1 — project foundation (implemented)**
   - Load YAML configuration.
   - Create runtime folders.
   - Configure deterministic seeds and logs.
   - Verify placeholder imports.
2. **Stage 2 — data pipeline (implemented)**
   - Build a Shanghai Yangpu instance from approved public/custom inputs.
   - Always retain a deterministic fallback instance that needs no network.
   - For formal AutoDL experiments, use `data_mode:
     original_scale_real_transit` so scale/settings inherit the previous
     eBus-Drone article while real transit stop/timetable CSVs are used where
     available.
3. **Stage 3 — event-driven MDP (implemented)**
   - Consume a validated Stage 2 instance.
   - Implement event/state/action/reward behavior and invariant tests.
4. **Later stages — learning and RLAIF (planned)**
   - Add learning only after simulator tests and baselines are stable.

## Experiment procedure

1. Copy or select a version-controlled configuration.
2. Set and record a seed.
3. Run the relevant smoke and unit tests.
4. Store logs in `logs/`, outputs in `outputs/`, and checkpoints in `checkpoints/`.
5. Record the command, commit, environment, result, and observations in
   `docs/EXPERIMENT_LOG.md` or a generated experiment manifest.
6. Do not silently overwrite an existing experiment directory.

## Stage 1 verification

```bash
python -m experiments.smoke_test_project --config configs/shanghai_small.yaml
```

This check is local-only and performs no downloads.


## Stage 3 verification

```bash
python -m experiments.smoke_test_environment --config configs/shanghai_small.yaml
python -m pytest -q tests/test_stage3_environment.py
```

The smoke gate forces the deterministic Stage 2 fallback build, completes an
episode with the first-feasible baseline, and checks simulator invariants after
every assignment and bus decision.

## Original-scale real-transit verification

```bash
python -m experiments.smoke_test_original_scale_real_transit_data --config configs/original_scale_real_transit.yaml
python -m experiments.smoke_test_original_scale_real_transit_env --config configs/original_scale_real_transit.yaml
python -m experiments.smoke_test_original_scale_real_transit_rlaif --config configs/original_scale_real_transit.yaml
```

These checks use fixture transit CSVs and temporary directories. They verify
interfaces, provenance, scale reports, timetable-driven bus events, and
source-aware prompts without claiming the fixture data are real research inputs.
