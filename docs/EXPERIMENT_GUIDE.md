# Experiment Guide: Preference-Aligned Async MAPPO Parcel Delivery

This repository targets dynamic truck--electric-bus--drone parcel delivery with three assignment modes: TD (truck direct), TBD (truck--electric-bus--drone), and TLD (truck--locker/station--drone).

## Scope and model alignment

- The environment is event-driven and asynchronous: assignment, truck, bus, and station agents act only when their own decision events occur.
- Async MAPPO uses type-specific actors with parameter sharing within each agent type, a centralized critic, decentralized execution, candidate-action scoring, action masks, and event-time transition storage.
- RLAIF reward shaping is assignment-agent-only in this codebase. `RLAIF_AGENT_TYPES = {"assignment"}` means truck, bus, and station rewards are never modified by the learned reward model.
- The station agent chooses only `dispatch_drone` or `idle`. Battery recharging is handled by environment dynamics after drone dispatch/return, not by a learned station-agent charging action.
- API scoring is an offline preference-data generation step. MAPPO training reads local preference-derived reward-model checkpoints and does not call an API in real time.

## Commands

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Environment smoke test:

```bash
python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback
```

3. Collect assignment states:

```bash
python -m experiments.collect_assignment_states --config configs/shanghai_small.yaml --output data/preference/assignment_states.jsonl --fallback
```

4. Build AI preference prompts:

```bash
python -m experiments.build_ai_preference_prompts --config configs/rlaif_preferences.yaml
```

5. Generate API-based preferences offline. Keys and endpoints must come only from environment variables:

```bash
export RLAIF_API_KEY=...
export RLAIF_API_BASE_URL=...
export RLAIF_MODEL_NAME=...
python -m experiments.build_ai_preferences --config configs/rlaif_preferences.yaml
```

6. Train the reward model:

```bash
python -m experiments.train_reward_model --config configs/train_reward_model.yaml
```

7. Train async MAPPO without RLAIF:

```bash
python -m experiments.train_mappo_async --config configs/train_mappo_async.yaml
```

8. Train async MAPPO with assignment-only RLAIF: set `rlaif.enabled: true`, set `rlaif.reward_model_checkpoint`, and optionally enable `rlaif.validation`. If the model is missing or invalid, safe defaults fall back to environment reward and log a warning.

9. Evaluate a checkpoint:

```bash
python -m experiments.evaluate_mappo_async --config configs/train_mappo_async.yaml --checkpoint results/checkpoints/mappo_async.pt
```

10. Ablation and sensitivity suites, when checkpoints exist:

```bash
python -m experiments.run_ablation --config configs/ablation.yaml
python -m experiments.run_sensitivity --config configs/sensitivity.yaml
```

## Phase 7 MAPPO reward-scale artifact

Before full environment-only MAPPO training, run `python -m experiments.estimate_reward_reference_scales --config configs/shanghai_small.yaml`. The command collects seeded preliminary policy rollouts, estimates robust nonzero component scales, stores project-specific component names, and writes a frozen versioned JSON artifact for training/checkpoint provenance.

### Phase 8 smoke pipeline

Replay fixtures may be used only to validate the pipeline mechanics, not to claim final reward-model quality. A Phase 8 smoke run should: collect four-agent preference states, build prompts, validate replay labels, build grouped splits, train smoke reward models, evaluate them, load all four wrappers, and run one RLAIF-MAPPO rollout. Formal experiments must use real external or validated replay labels and formal sample/accuracy thresholds.

## Phase 9 paper smoke workflow

Run `python -m experiments.smoke_test_experiments` before launching formal experiments. This tiny workflow generates one scenario per bank, validates the policy matrix, validates ablation and sensitivity declarations, runs a paired benchmark smoke, and aggregates smoke records. These smoke artifacts are not final results.

Formal studies should use frozen test-bank scenarios for all methods, train separate policy checkpoints for each reward setting or ablation, and record an artifact manifest beside each run.

### Fix Phase 2 bus diagnostics

Environment smoke tests report actual bus runtime metrics including scheduled trips started/completed, freight and non-freight completions, ordinary/integrated stops visited, ordinary-stop boardings/alightings, segment count, propulsion/relocation/charging energy, and minimum physical-bus SoC. These values are runtime diagnostics from the stop-by-stop event chain, not timetable-row inferences.

## Fix Phase 6 formal evaluation integrity

Formal evaluation now uses frozen scenario-bank manifests. All methods share identical scenario artifacts, and paired comparisons validate scenario ID, instance hash, scenario-manifest hash, and exogenous artifact hashes before comparison. Environment MAPPO, assignment-only RLAIF-MAPPO, and full RLAIF-MAPPO are separate formal method identities with separate policy checkpoints. Full RLAIF evaluation requires four agent-specific reward checkpoints loaded through `RewardRegistry`; assignment-only RLAIF enables only the assignment reward model. Reward models do not select evaluation actions; they validate lineage and score selected transitions for decomposition only.

Formal metrics are fail-closed: missing instrumentation is missing, not zero. Legitimate zero values require an instrumented source and explicit legitimate-zero provenance. Ablations that require retraining require separate checkpoints and actual configuration differences. Sensitivity experiments distinguish fixed-policy robustness from retrained-policy sensitivity and do not aggregate the two modes together by default.

This infrastructure does not claim that the final 100-scenario, three-seed paper benchmark has been executed; formal readiness remains blocked until final frozen banks, trained policies, and validated formal reward checkpoints exist.

## Fix Phase 7 readiness note

Status: pilot-validated for the diagnostic readiness pilot; blocked for formal RLAIF until all four final formal reward checkpoints and manifests validate. Diagnostic and smoke artifacts are not experiment-validated formal artifacts.

## Step 8a Validation Status

This step corrects environment semantics and formal configuration gates only. It
must not be interpreted as final formal-training readiness: final reward-scale
estimation and a real end-to-end readiness pilot remain later-step work.

### Step 6A reward-scale protocol

Use `configs/paper/reward_scale_estimation.yaml` as the formal protocol template. It requires a frozen train split bank and writes the final artifact only when run later by the user. Diagnostic preparation and estimation write under `results/diagnostic/reward_scales/`, which is ignored and must not be committed.
