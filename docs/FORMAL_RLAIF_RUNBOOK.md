# Formal Four-Agent RLAIF Runbook

This runbook describes the fail-closed operational sequence for the formal four-agent RLAIF experiment pipeline. Runtime artifacts are written under `results/formal/` and remain untracked by Git.

## Step 1: configure API

Configure the OpenAI-compatible evaluator before generating formal labels:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."
export OPENAI_MODEL="..."
```

The preference generator refuses to create preference datasets if any required API setting is missing. It also rejects malformed evaluator responses and records failed requests under `results/formal/rlaif/failed/`.

## Step 2: prepare formal scenario banks and reward scale

```bash
python -m experiments.prepare_formal_experiment_inputs \
  --output-root results/formal \
  --resume
```

Expected outputs include the frozen train/validation/test scenario-bank manifests and the final reward-scale artifact under `results/formal/`. On a first run, use normal creation or `--force` to intentionally rebuild all requested formal inputs. On interrupted runs, use `--resume` to reuse only validated compatible scenario banks and the reward-scale estimator progress. `--resume` and `--force` are mutually exclusive.

All scenario identities in formal configs and lineage use the scenario-bank manifest internal canonical `bank_hash`. The manifest-file SHA is recorded separately as `manifest_file_hash` for file-integrity metadata only and must not substitute for `bank_hash`.

## Step 3: generate preferences and train four reward models

```bash
python -m experiments.prepare_formal_rlaif_artifacts \
  --config configs/paper/rlaif_preference_generation.yaml \
  --output-root results/formal/rlaif \
  --resume
```

This command validates the formal train bank, checks API configuration, generates or resumes preference collection when datasets are absent/stale/incomplete, validates scenario-grouped split isolation, trains four independent reward models, validates checkpoints, and writes complete MAPPO runtime configs.

Expected preference outputs:

```text
results/formal/rlaif/preferences/preference_assignment.jsonl
results/formal/rlaif/preferences/preference_truck.jsonl
results/formal/rlaif/preferences/preference_bus.jsonl
results/formal/rlaif/preferences/preference_station.jsonl
results/formal/rlaif/preference_manifest.json
results/formal/rlaif/failed/
results/formal/rlaif/evaluator_cache/
```

Expected reward and config outputs:

```text
results/formal/reward_models/reward_assignment.pt
results/formal/reward_models/reward_truck.pt
results/formal/reward_models/reward_bus.pt
results/formal/reward_models/reward_station.pt
results/formal/rlaif/formal_rlaif_artifact_manifest.json
results/formal/configs/mappo_rlaif_assignment.yaml
results/formal/configs/mappo_rlaif_all.yaml
```

The command fails closed if target usable-label counts are not met, if bus preferences do not cover both bus decision events, if split leakage is detected, if reward checkpoints are invalid, or if resolved configs still contain placeholders.

## Step 4: train policies

For seeds `1`, `2`, and `3`, run:

```bash
python -m experiments.train_assignment_ppo \
  --config results/formal/configs/assignment_ppo.yaml \
  --seed <SEED>

python -m experiments.train_mappo_async \
  --config results/formal/configs/mappo_env.yaml \
  --seed <SEED>

python -m experiments.train_mappo_async \
  --config results/formal/configs/mappo_rlaif_assignment.yaml \
  --seed <SEED>

python -m experiments.train_mappo_async \
  --config results/formal/configs/mappo_rlaif_all.yaml \
  --seed <SEED>
```

Expected outputs are three policy checkpoints per learned method. The benchmark resolver requires all seeds for `assignment_ppo`, `mappo_env`, `mappo_rlaif_assignment`, and `mappo_rlaif_all`; it will not silently omit missing methods or seeds.

## Step 5: resolve and run benchmark

```bash
python -m experiments.resolve_formal_benchmark \
  --template configs/paper/benchmark.yaml \
  --artifact-root results/formal \
  --output results/formal/configs/benchmark.yaml

python -m experiments.run_paper_benchmark \
  --config results/formal/configs/benchmark.yaml
```

The benchmark resolver inspects actual policy and reward checkpoints, validates lineage metadata, binds checkpoint paths and hashes for seeds `1`, `2`, and `3`, injects the test scenario-bank hash when present, and writes `results/formal/configs/benchmark.yaml` only after all required artifacts validate. Missing or incompatible policy/reward checkpoints return nonzero and list the exact missing method/seed or invalid reward agent.
