# Hard parcel-locker formal re-experiment

## Publication validity and design

The previous environment checked only physical locker load during assignment, so
several in-transit parcels could commit the same capacity and overflow on arrival.
Those MAPPO/RLAIF-MAPPO checkpoints and all formal results produced with them are
invalid for publication. They must be archived, never overwritten or reused.

The corrected state separates physical `locker_load_kg` from inbound
`locker_reserved_kg`. TBD/TLD assignment atomically reserves capacity; station
arrival converts exactly one reservation to physical occupancy; cancellation
releases inbound capacity; drone dispatch strictly releases physical occupancy.
TD never reserves. Every mutation is checked and invariant failures are fatal.

Assignment state and candidate vectors expose remaining capacity, occupancy, and
post-assignment load, so their semantics changed without changing dimensions.
Consequently the assignment reward model consumes changed inputs. Observation and
candidate schema version 4 deliberately invalidates the old checkpoint; regenerate
assignment-only preference data and the assignment reward model. Formal fallback
remains forbidden. Policy checkpoints are likewise retrained under the corrected
environment.

## Reproducible workflow

The training/test scenario banks remain reusable inputs, subject to their normal
manifest hash validation. Outputs use
`results/formal/hard_locker_rerun_<commit-prefix>` and are never overwritten.
Every generated artifact belongs below that commit-specific root: preferences, reward-model checkpoints and manifests, reward scales, resolved and frozen configs, six training runs, readiness evidence, benchmark rows, paired analysis, provenance, and phase markers. No phase writes into the previous invalid result directories. Reward-scale JSON and hashes are generated and never edited manually.

Inspect the fail-closed plan, then execute/resume phases:

```bash
python -m experiments.run_hard_locker_reexperiment --through 9
python -m experiments.run_hard_locker_reexperiment --through 9 --execute
python -m experiments.run_hard_locker_reexperiment --through 9 --execute --resume
```

Exact phase commands are emitted as JSON. The key standalone commands are:

```bash
# verification and smoke
python -m pytest -q tests/test_hard_locker_capacity.py
python -m experiments.smoke_test_environment --config configs/shanghai_small.yaml

# preference/reward-model regeneration (assignment-only)
ROOT=results/formal/hard_locker_rerun_<commit-prefix>
python -m experiments.generate_formal_multiagent_preferences --config configs/paper/rlaif_preference_generation.yaml --output-root $ROOT/preferences
python -m experiments.train_multi_agent_reward_models --preferences $ROOT/preferences --config $ROOT/resolved_configs/train_reward_assignment.yaml --agent assignment --output $ROOT/reward_models/reward_assignment.pt

# scale regeneration; structurally-zero locker_overflow retains a documented
# positive denominator, not an empirical percentile
python -m experiments.estimate_reward_reference_scales --scenario-bank results/formal/scenario_banks/train/manifest.json --config configs/paper/reward_scale_estimation.yaml --output $ROOT/reward_scales/final_reward_reference_scales.json

# readiness, 600-row common-scenario evaluation, paired analysis
python -m experiments.validate_formal_experiment_readiness --config $ROOT/resolved_configs/benchmark.yaml --strict
python -m experiments.run_paper_benchmark --config $ROOT/resolved_configs/benchmark.yaml --output-root $ROOT/benchmark
python -m experiments.aggregate_paper_results --input $ROOT/benchmark --output $ROOT/paired_analysis.json
```

Phases 5 and 6 invoke the real MAPPO trainer for seeds 1, 2, and 3. Phase 7
requires six distinct loadable trained-actor checkpoints and valid policy, reward
model, reward-scale, and scenario-bank lineage. Phase 8 requires exactly 600
successful rows over the same 100 scenario IDs and rejects fallback, mask,
overflow, residual reservation, or invariant failures. Phase 9 pairs by scenario,
averages the three training seeds per method, and treats locker metrics only as
feasibility checks.

Successful verification prints `HARD_LOCKER_CAPACITY_SMOKE_VALID`; the formal pilot
prints `HARD_LOCKER_FORMAL_GATE_VALID`. Archive old results outside the new root or
mark them `INVALID_PRE_HARD_LOCKER`; never copy their checkpoints into a resume
directory. Resume accepts only phase markers under the current commit-specific root.

Formal training starts only after the follow-up PR tests pass. After merge, the exact GPU-server launch is:

```bash
python -m experiments.run_hard_locker_reexperiment --through 9 --execute
```
