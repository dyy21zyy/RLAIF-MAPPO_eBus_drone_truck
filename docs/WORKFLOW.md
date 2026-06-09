# Development Workflow

## Stage gates

Each stage must pass its own offline smoke test before downstream work begins.

1. **Stage 1 — project foundation (implemented)**
   - Load YAML configuration.
   - Create runtime folders.
   - Configure deterministic seeds and logs.
   - Verify placeholder imports.
2. **Stage 2 — data pipeline (planned)**
   - Build a Shanghai Yangpu instance from approved public/custom inputs.
   - Always retain a deterministic fallback instance that needs no network.
3. **Stage 3 — event-driven MDP (planned)**
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
