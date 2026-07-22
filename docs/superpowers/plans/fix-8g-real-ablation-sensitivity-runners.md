# Step 6B Plan: real ablation and sensitivity runners

Starting main SHA: `1faa818a59499b70fb0b9e74ea03524128298fd4` (local checkout; remote `origin` unavailable in this container).

## Scope

Implement source-controlled matrix schemas and runners that execute real orchestration rather than YAML-only validation. Formal paper experiments are not run in this PR.

## Matrix schemas

- `configs/paper/ablation_matrix.yaml` declares formal ablation variants, baseline ID, training seeds, frozen train/validation/test bank manifests, execution modes, training configs, and benchmark method IDs.
- `configs/paper/sensitivity_matrix.yaml` declares fixed-policy robustness and retrained-policy sensitivity protocols, factors, values, intended config paths, master seeds, and scenario generation requirements.

## Training-required vs evaluation-only

- `retrain_and_evaluate` jobs resolve one MAPPO training config per variant/factor value and seed, invoke `experiments.train_mappo_async`, validate checkpoint presence and hash, then evaluate.
- `fixed_policy_evaluate` jobs require a declared checkpoint and skip training.

## Sensitivity separation

- Fixed-policy robustness preserves one checkpoint hash across parameter values.
- Retrained-policy sensitivity writes separate training roots and checkpoint paths for each value and seed.

## Scenario-family generation

- Scenario families are modeled by `evaluation.scenario_family`, preserving master seed and stochastic seed tuple identity across factor values.
- Diagnostic matrices generate tiny train/validation/test banks under `results/diagnostic/experiment_matrices`.

## Resolved configs and job execution

- Runners write resolved training and benchmark configs under runtime `resolved_configs/` only.
- Commands are invoked with `subprocess.run(..., capture_output=True)` through a checked helper; nonzero return codes become failed jobs.

## Resume identity

- Jobs compute canonical JSON identity from experiment kind, variant/protocol/factor/value, seed, config hash, bank hash, policy/reward lineage, reward scale, and code compatibility token.
- Resume skips only prior terminal successful rows with exact identity hash.

## Failure handling

- Failed training/evaluation jobs remain in `job_results.jsonl`, `job_results.csv`, and `failure_report.json` with stage, command, return code, exception type, reason, and runtime when available.
- Validation-only returns `validated`, never experiment success.

## Paired aggregation

- `evaluation.experiment_aggregation` groups only compatible rows and computes descriptive statistics and paired differences before averaging.
- Diagnostic/formal rows, fixed/retrained sensitivity protocols, assignment-only/all-agent RLAIF scopes, failed rows, and metric-schema mismatches are separated.

## Runtime artifact isolation and gates

- Diagnostic outputs remain under ignored `results/diagnostic/*` roots.
- Formal outputs remain under ignored `results/formal/ablation` and `results/formal/sensitivity`.
- Staging uses explicit whitelists; generated-artifact, binary-extension, and Git binary-detection gates run before PR creation.
