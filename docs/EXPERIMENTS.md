# Experiment Catalog

> Stages 1–8 have passed their dependency-light code gates. PyTorch training and
> final benchmark execution remain deferred runtime work.

## Naming convention

Use `<stage>_<purpose>_<city>_<seed>_<timestamp>` for artifact directories.
Store the resolved configuration and a metadata manifest with every run.

## Implemented experiment families and deferred runtime work

| Family | Config | Status | Primary purpose |
| --- | --- | --- | --- |
| Project smoke | `configs/shanghai_small.yaml` | Implemented | Validate Stage 1 foundation |
| Data pipeline smoke | `configs/shanghai_small.yaml` | Stage 2 implemented | Validate fallback instance |
| Original-scale real-transit smoke | `configs/original_scale_real_transit.yaml` | Implemented | Validate source-aware real-transit/inherited data mode with fixtures |
| Environment smoke | `configs/shanghai_small.yaml` | Stage 3 implemented and hardened | Complete one valid episode |
| Reward model | `configs/train_reward_model.yaml` | Stage 5 Code Gate complete; runtime deferred | Fit preference reward |
| Assignment PPO | `configs/train_assignment_ppo.yaml` | Stage 6 Code Gate complete; runtime training deferred | Assignment-only policy |
| MAPPO | `configs/train_mappo_async.yaml` | Stage 7 Code Gate complete; runtime training deferred | Cooperative assignment/bus policy |
| Benchmark, ablation, sensitivity | `configs/{experiments,ablation,sensitivity}.yaml` | Stage 8 experiment framework implemented; final experiments deferred | Fair evaluation and aggregation |

## Minimum reporting checklist

- Commit and clean/dirty status
- Fully resolved configuration
- Seed set
- Dataset/instance identifier
- Exact command
- Runtime and hardware
- Success/failure status
- Metrics with definitions
- Artifact paths
- Known limitations

## Stage 6 assignment PPO commands

```bash
# Dependency-light code smoke test; uses rlaif.enabled=false and a temporary directory.
python -m experiments.smoke_test_assignment_ppo

# Training (requires PyTorch).
python -m experiments.train_assignment_ppo --config configs/train_assignment_ppo.yaml

# Deterministic evaluation with the same fixed bus baseline (requires PyTorch).
python -m experiments.evaluate_assignment_ppo --config configs/train_assignment_ppo.yaml --checkpoint results/checkpoints/assignment_ppo.pt
```

The actor and critic are separate `[256, 256]` ReLU MLPs. The actor samples a
masked categorical distribution over `1 + 2H` assignment actions; the critic is a
local state-value model, not a centralized critic. PPO uses normalized GAE,
clipped likelihood ratios, MSE value loss, entropy regularization, and gradient
clipping. Rollout storage contains assignment transitions only.

The default config disables RLAIF, so smoke testing does not require Stage 5
runtime output. Enabling RLAIF requires a valid trained checkpoint and never
activates a rule fallback. Training CSVs, evaluation JSON, and checkpoints under
`results/` are runtime artifacts and are not committed. These commands are code
and runtime checks, not Stage 8 experiments or final RLAIF-enabled validation.

## Stage 7 code-gate runs (not Stage 8 experiments)

Use `configs/train_mappo_async.yaml` for asynchronous MAPPO implementation checks:

```bash
python -m experiments.smoke_test_mappo_async
python -m experiments.train_mappo_async --config configs/train_mappo_async.yaml
python -m experiments.evaluate_mappo_async --config configs/train_mappo_async.yaml --checkpoint results/checkpoints/mappo_async.pt
```

Training logs the assignment and bus decision counts and separate policy/entropy/KL
metrics, plus shared critic loss and delivery, lateness, charging, delay, overload,
and locker metrics. Evaluation uses deterministic masked actions. Runtime files go
under ignored `results/` paths. The smoke test uses temporary paths, one rollout
episode, one PPO epoch, disabled RLAIF, and skips cleanly without PyTorch.

These commands are code and interface validation only. Do not report them as final
Stage 8 comparisons. In particular, do not enable RLAIF unless the Stage 5 Runtime
Gate has produced and validated the configured `reward_model.pt`.

## Stage 9 four-agent code-gate runs (not final experiments)

Stage 9 four-agent asynchronous MAPPO uses the same `configs/train_mappo_async.yaml`
entry point, but the checkpoint metadata is `stage: 9` and
`algorithm: four_agent_asynchronous_mappo`. The smoke gate validates assignment,
truck, bus, and station transitions, candidate actions, candidate features, and
action masks.

Training logs four decision counts, four policy-loss/entropy/KL/clip-fraction
families, shared critic loss, and delivery/resource metrics. These outputs are
code-gate evidence only. They are not benchmark, ablation, sensitivity, or
paper-ready RLAIF-enabled experiment results.


# Stage 8 experiment framework

Stage 8 implements experiment code and dependency-light validation only. Its generated files are **not paper-ready results**, and no performance claim should be inferred from smoke runs.

## Commands

```bash
python -m experiments.smoke_test_experiments
python -m experiments.run_baselines --config configs/experiments.yaml
python -m experiments.run_benchmark --config configs/experiments.yaml
python -m experiments.run_ablation --config configs/ablation.yaml
python -m experiments.run_sensitivity --config configs/sensitivity.yaml
python -m experiments.aggregate_results --input results/raw --output results/summary
```

The smoke test builds one deterministic Stage 2 fallback instance, uses one seed, runs `truck_only`, `random_feasible`, `bus_drone_only`, `truck_drone`, and `rule_based`, explicitly skips an unavailable learned checkpoint, and aggregates temporary output. It does not need PyTorch or `reward_model.pt` for `rlaif_enabled=false` methods.

## Fairness and outputs

Every method in one benchmark receives the same Stage 2 instance and seed list. Per-episode JSON is written below `<output_dir>/raw/<method>/`, while `<output_dir>/episodes.csv` and `<output_dir>/summary/{summary_metrics.csv,summary_metrics.json,method_status.csv}` are plotting/table inputs. The entire `results/` tree is ignored by Git.

Missing Stage 6/7 checkpoints produce `skipped_missing_checkpoint`; missing PyTorch produces `skipped_missing_dependency`. No heuristic is silently substituted. An RLAIF method additionally requires a valid Stage 5 `reward_model.pt`; disabled RLAIF never loads that file. Invalid present checkpoints fail clearly rather than producing fabricated rewards.

Ablations are declarations of checkpoint-backed variants. Sensitivity dimensions are config-driven and smoke mode deliberately evaluates only a tiny subset. Full benchmark, ablation, and sensitivity execution remains deferred until PyTorch is available and valid Stage 5, Stage 6, and Stage 7 checkpoints have been trained in the proper runtime environment.

The rule-based policy is an interpretable comparison baseline only. Reusing it for preference labels or reward-model supervision would create circular, fabricated RLAIF evidence and is prohibited.

## Original-scale real-transit setting

The formal data mode is:

```yaml
data_mode: original_scale_real_transit
```

It preserves the scale of the previous eBus-Drone article while adding the truck
feeder layer. Bus stops, trips, stop sequences, and stop-level timetables come
from real transit CSVs when available. Missing real stop_times may be synthesized
only when explicitly allowed, and then the provenance is `original_ebus_drone`,
not `real_transit_data`.

Smoke commands:

```bash
python -m experiments.smoke_test_original_scale_real_transit_data --config configs/original_scale_real_transit.yaml
python -m experiments.smoke_test_original_scale_real_transit_env --config configs/original_scale_real_transit.yaml
python -m experiments.smoke_test_original_scale_real_transit_rlaif --config configs/original_scale_real_transit.yaml
```

Those commands use committed fixture CSVs and temporary output. They are code
validation artifacts, not final data collection or paper-ready experiments.

## Phase 9 formal paper experiment framework

Phase 9 adds a reproducible framework for formal training, benchmark, ablation, and sensitivity studies. The checked-in configuration files under `configs/paper/` define the protocol only; smoke outputs are engineering checks and must not be reported as final paper results.

* Scenario banks are generated by `python -m experiments.generate_scenario_banks --config configs/paper/scenario_banks.yaml` into disjoint `train`, `validation`, and `test` banks. Each bank writes a manifest with scenario IDs, seed tuples, configuration hashes, artifact hashes, size, and schema version.
* Policy matrix validation is provided by `python -m experiments.train_policy_matrix --config configs/paper/benchmark.yaml --validate-only`. Environment-reward MAPPO, assignment-only RLAIF-MAPPO, and full multi-agent RLAIF-MAPPO must use separate checkpoints.
* Paper benchmark, ablation, and sensitivity entry points are `experiments.run_paper_benchmark`, `experiments.run_paper_ablation`, and `experiments.run_paper_sensitivity`. Unavailable learned methods are explicit skips, never heuristic substitutions.
* Paired evaluation joins by `scenario_id` and rejects comparisons with mismatched scenario sets.
* Sensitivity studies keep fixed-policy robustness and retrained-policy sensitivity in separate tables.
* Aggregation reports mean, standard deviation, median, reproducible bootstrap confidence intervals, success counts, failure counts, and skip counts; skipped runs are never treated as zero.
* Artifact manifests capture git state, resolved configuration, scenario/preference/reward/policy hashes, seeds, Python/PyTorch versions, hardware, timestamps, runtime, status, and failure reason.

## Fix Phase 6 formal evaluation integrity

Formal evaluation now uses frozen scenario-bank manifests. All methods share identical scenario artifacts, and paired comparisons validate scenario ID, instance hash, scenario-manifest hash, and exogenous artifact hashes before comparison. Environment MAPPO, assignment-only RLAIF-MAPPO, and full RLAIF-MAPPO are separate formal method identities with separate policy checkpoints. Full RLAIF evaluation requires four agent-specific reward checkpoints loaded through `RewardRegistry`; assignment-only RLAIF enables only the assignment reward model. Reward models do not select evaluation actions; they validate lineage and score selected transitions for decomposition only.

Formal metrics are fail-closed: missing instrumentation is missing, not zero. Legitimate zero values require an instrumented source and explicit legitimate-zero provenance. Ablations that require retraining require separate checkpoints and actual configuration differences. Sensitivity experiments distinguish fixed-policy robustness from retrained-policy sensitivity and do not aggregate the two modes together by default.

This infrastructure does not claim that the final 100-scenario, three-seed paper benchmark has been executed; formal readiness remains blocked until final frozen banks, trained policies, and validated formal reward checkpoints exist.

## Fix Phase 7 readiness note

Status: pilot-validated for the diagnostic readiness pilot; blocked for formal RLAIF until all four final formal reward checkpoints and manifests validate. Diagnostic and smoke artifacts are not experiment-validated formal artifacts.

## Diagnostic pre-formal gate

The diagnostic pre-formal gate is a small, real integration workload and is explicitly publication-ineligible (`publication_eligible=false`, `run_classification=diagnostic`). Generate its local-only inputs with `python -m experiments.prepare_diagnostic_preformal_gate --output-root results/diagnostic/preformal_gate --force`. The helper writes scenario-bank manifests, reward-scale metadata, diagnostic reward checkpoints, and resolved configs only below `results/diagnostic/preformal_gate/`, which is ignored by Git.

Use `python -m experiments.run_preformal_gate --config results/diagnostic/preformal_gate/preformal_gate.resolved.yaml --output-root results/diagnostic/preformal_gate/run --validate-only` for report-only validation, then omit `--validate-only` and add `--continue-on-error` to execute the diagnostic gate. Use `--resume --continue-on-error` to exercise identity-based resume behavior.

Strict pre-formal validation is different: `python -m experiments.run_preformal_gate --config configs/preformal/preformal_gate.template.yaml --output-root results/preformal/gate --validate-only --strict` must honestly report blockers while formal artifacts and hashes are unresolved. The diagnostic pre-formal workflow has been executed to validate the integration path. Final formal readiness still requires real formal scenario banks, reward-scale artifacts, validated reward checkpoints, and a passing strict pre-formal run.
