# Experiment Readiness Report

Fix Phase 7 adds an instrumented diagnostic pilot. The pilot status is **pilot-validated** for environment/MAPPO diagnostics when `python -m experiments.run_readiness_pilot ...` passes, and **blocked** for formal RLAIF if the four final formal reward checkpoints are absent or invalid.

The diagnostic pilot is not a formal experiment and must not be used as paper result evidence. No component is marked experiment-validated by this phase.

Expected honest status for this repository without final formal reward checkpoints: `ENV_MAPPO_READY_RLAIF_BLOCKED`.

## Reward scale readiness blocker

Formal MAPPO and benchmark configs now require `outputs/reward_reference_scales_v1.json`, its real artifact hash, and the real frozen training-bank hash. The formal artifact was not generated in Step 6A; readiness remains blocked until the final train bank is frozen and the user runs formal reward-scale estimation.
