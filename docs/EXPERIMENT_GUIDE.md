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
