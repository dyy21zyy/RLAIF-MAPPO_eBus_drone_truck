# fix-8m full four-agent RLAIF go-live plan

Starting local SHA: `a80706a45fecfcc06b884192dfae99bf20d56762`.

Remote precondition note: this worktree has no configured `origin`, so `git fetch origin` / `git pull --ff-only origin main` cannot complete in this environment. Work proceeds from the current checked-out repository state and records the blocker in the PR evidence.

## Inspected canonical implementation

- Runtime event contract: `training/event_schema.py`.
- Runtime RLAIF registry and selected-transition scoring: `rlaif/reward_registry.py`, `rlaif/runtime_agent_reward_model.py`, `training/reward_model_wrapper.py`.
- Formal input preparation: `experiments/prepare_formal_experiment_inputs.py`.
- Async training/benchmark entry points: `experiments/train_mappo_async.py`, `experiments/run_paper_benchmark.py`.
- Preference schemas and reward datasets: `rlaif/preference_schema_v3.py`, `rlaif/reward_model_dataset.py`, `rlaif/grouped_split.py`.
- Reward-model training: `experiments/train_multi_agent_reward_models.py`, `rlaif/reward_model_trainer.py`.
- Existing evaluator adapter: `rlaif/ai_evaluator.py`.

## Implementation plan

1. Add a source-controlled formal four-agent preference-generation config with per-agent event coverage, counts, split policy, evaluator environment-variable names, retry/cache policy, and output paths.
2. Add a resumable formal artifact-preparation entry point that orchestrates formal precondition validation, evaluator credential/test-request checks, formal scenario lineage checks, per-agent preference data validation, per-agent reward-model training, strict checkpoint validation, manifest generation, and runtime full-RLAIF config materialization.
3. Reuse the canonical reward-model trainer/checkpoint loader and runtime `RewardRegistry`; do not add action-selection or candidate-reranking behavior.
4. Add focused tests for formal evaluator parsing/credentials/cache identity and four-agent artifact gating without creating runtime checkpoints or datasets in source control.
5. Run formal scenario/reward-scale prep, then the formal artifact command. If evaluator credentials are absent, stop with a nonzero error and no fake labels/checkpoints.
6. Only after real credentials and labels are available, train the four checkpoints, generate `results/formal/configs/mappo_rlaif_all.yaml`, run reduced full-RLAIF training and benchmark, and record machine-produced evidence.

## Non-negotiable artifact policy

Runtime outputs under `results/formal/` are intentionally ignored and must not be committed: scenario banks, preference JSONL, evaluator cache, `.pt` checkpoints, policy checkpoints, logs, resolved runtime configs, and benchmark outputs.
