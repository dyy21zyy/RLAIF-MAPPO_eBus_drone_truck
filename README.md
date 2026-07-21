# RLAIF-MAPPO for Dynamic Multimodal Parcel Delivery

This repository is being developed in explicit stages for parcel assignment and
scheduling across trucks, electric buses, integrated stations, and drones.

## Current status: Stage 8 framework implemented; runtime experiments deferred

Stage 2 provides a reproducible Shanghai instance data pipeline with two road
network modes:

- **Full mode:** attempts an OpenStreetMap drivable-network download through
  `osmnx` when that optional package is installed, and automatically falls back
  if the download fails.
- **Fallback mode:** uses a deterministic 25-node directed grid, a generated
  20-stop corridor route, a synthetic timetable and parcels, and requires no
  internet access.

The repository also includes the source-aware `original_scale_real_transit` data
mode. This project extends the previous Electric Bus-Drone paper by adding a
truck feeder layer; the formal data design therefore inherits the previous
eBus-Drone scale and system parameters wherever available, uses real transit
data where available, and records provenance for every important inherited,
real, explicit truck-extension, or fallback-only field. Missing real data are
not claimed as real.

Real transit data are used where available; otherwise missing fields are filled
from the previous eBus-Drone setting or an explicitly documented truck-extension
default.

Stage 3 provides the hardened deterministic event-driven assignment and
electric-bus charging MDP. Stage 4 provides assignment-state collection,
objective candidate features, versioned pairwise AI prompts, and
offline/API/replay preference validation. Stages 5–7 implement the reward-model,
assignment-PPO, and asynchronous-MAPPO code gates; their runtime training is
deferred. Stage 8 implements the experiment framework, while final experiments
remain deferred.

## Repository layout

```text
configs/        Scenario, training, and experiment configurations
data/           Raw inputs and ignored generated instances
data_pipeline/  Stage 2 road, bus, facility, parcel, matrix, and instance builders
checkpoints/    Ignored runtime model artifacts
docs/           Design, guardrail, and experiment documentation
envs/           Stage 3 event-driven assignment and bus environment
experiments/    Offline gates, training/evaluation, and Stage 8 runners
logs/           Runtime logs
models/         Future model placeholder
outputs/        Future experiment outputs
rlaif/          Stage 4 preference workflow and Stage 5 reward-model training
tests/          Stage 1 through Stage 8 regression and hardening tests
training/       Future optimization-loop placeholder
utils/          Config, logging, and reproducibility utilities
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`osmnx` is optional and is needed only for full OSM mode. The fallback build and
smoke test use standard-library pipeline code and can run without network access.
NumPy is declared for downstream consumers of the generated `.npy` matrices,
while Stage 2 also includes a dependency-light NumPy-format writer.

## Build the Stage 2 fallback instance

From the repository root, run:

```bash
python -m data_pipeline.build_instance --config configs/shanghai_small.yaml --fallback
```

The output is written to `data/processed/shanghai_yangpu/` and includes the road
graph and tables, bus tables and timetable, depot, integrated stations, parcels,
three distance matrices, and both `instance.yaml` and `instance.json` manifests.
Generated files are ignored by Git and can be safely rebuilt.

To attempt full OSM mode after installing `osmnx`, omit `--fallback`:

```bash
python -m data_pipeline.build_instance --config configs/shanghai_small.yaml
```

A user-provided route can be supplied with `--bus-route path/to/route.csv`. Its
required columns are `route_id`, `stop_id`, `stop_name`, `stop_sequence`, `lat`,
`lon`, `first_departure`, `last_departure`, and `headway_min`.

## Build original-scale real-transit data

Formal original-scale builds use:

```bash
python -m data_pipeline.build_instance --config configs/original_scale_real_transit.yaml
```

This mode expects real transit CSVs at the configured `data/raw/transit/real_*`
paths and writes `instance.json`, `data_provenance.json`, and
`scale_match_report.json` below `data/processed/<city_name>/`. Raw real transit
CSVs and generated processed data are ignored by Git.

Dependency-light smoke commands use tiny committed fixtures and temporary output
only:

```bash
python -m experiments.smoke_test_original_scale_real_transit_data --config configs/original_scale_real_transit.yaml
python -m experiments.smoke_test_original_scale_real_transit_env --config configs/original_scale_real_transit.yaml
python -m experiments.smoke_test_original_scale_real_transit_rlaif --config configs/original_scale_real_transit.yaml
```

See [docs/DATA_SETTING_TRACE.md](docs/DATA_SETTING_TRACE.md) and
[docs/RANDOMNESS_DESIGN.md](docs/RANDOMNESS_DESIGN.md).

## Run the Stage 3 environment

Build an instance as above, then initialize `DynamicDeliveryEnv` with its
`instance.json`. The dependency-light API follows Gymnasium reset/step return
signatures without requiring Gymnasium itself. For an end-to-end deterministic
episode, run:

```bash
python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback
```

See [docs/MDP_SPECIFICATION.md](docs/MDP_SPECIFICATION.md) for event, action,
transition, reward, and termination semantics.

## Build Stage 4 RLAIF data

```bash
python -m experiments.collect_assignment_states --config configs/shanghai_small.yaml --episodes 50 --output data/preference/assignment_states.jsonl --fallback
python -m experiments.build_ai_preference_prompts --states data/preference/assignment_states.jsonl --output data/preference/ai_preference_prompts.jsonl
python -m experiments.build_ai_preferences --mode offline --prompts data/preference/ai_preference_prompts.jsonl
python -m experiments.smoke_test_rlaif_data --config configs/shanghai_small.yaml --fallback
```

Offline mode deliberately writes no preference labels. For manual labels or a
configured external evaluator, see [docs/RLAIF_WORKFLOW.md](docs/RLAIF_WORKFLOW.md).


## Train and evaluate the Stage 5 reward model

The Stage 5 Code Gate is dependency-light and passes without PyTorch. The separate
Runtime Gate is deferred and must be run after producing valid Stage 4 API or replay
labels in a local/AutoDL environment with `torch>=2.0,<3.0`:

```bash
python -m experiments.smoke_test_reward_model
python -m experiments.train_reward_model --config configs/train_reward_model.yaml --data data/preference/ai_preferences.jsonl
python -m experiments.evaluate_reward_model --config configs/train_reward_model.yaml --checkpoint results/checkpoints/reward_model.pt
python -m pytest -q
```

When PyTorch is unavailable, Stage 5 runtime commands exit cleanly with an
installation message and PyTorch-dependent tests are skipped. This is not evidence
that a reward model has been trained or evaluated.

Training accepts only valid, usable pairwise labels and never creates or falls
back to rule labels. Missing, empty, or unusable preference input stops with an
actionable message. Checkpoints save feature normalization and training-score
mean/std. The current ten-label replay data is for pipeline validation only, not
final reward-model quality. See [docs/RLAIF_WORKFLOW.md](docs/RLAIF_WORKFLOW.md).

## Verification

```bash
python -m experiments.smoke_test_project --config configs/shanghai_small.yaml
python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback
python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback
python -m pytest -q
python -m compileall -q .
git diff --check
```

The Stage 2 smoke test validates the complete fallback artifact set. The Stage 3
smoke test builds that instance in a temporary directory, runs a complete
event-driven episode, and checks state invariants after every decision. Neither
smoke test requires a network request.

## Development boundaries

- Stage 1: foundation and documentation (complete).
- Stage 2: offline-capable Shanghai instance data pipeline (complete).
- Stage 3: event-driven MDP environment (implemented and hardened).
- Stage 4: RLAIF state/prompt collection and AI-label interface (complete).
- Stage 5: Code Gate complete; PyTorch Runtime Gate deferred.
- Stage 6: Code Gate complete; runtime training deferred.
  The bus uses a fixed baseline; Stage 6 contains no MAPPO or centralized critic.
- Stage 7: Code Gate complete; runtime training deferred.
  Final RLAIF-enabled experiments remain blocked until `reward_model.pt` has passed
  the deferred Stage 5 Runtime Gate.
- Stage 8: experiment framework implemented; final experiments deferred.

Final RLAIF-enabled runtime experiments require all of the following:

1. a working PyTorch environment;
2. completion of the Stage 5 Runtime Gate;
3. a valid trained `reward_model.pt`;
4. trained `assignment_ppo.pt` and/or `mappo_async.pt`; and
5. benchmark/ablation/sensitivity runs in the intended AutoDL experiment environment.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the staged workflow and
[docs/PITFALLS.md](docs/PITFALLS.md) for scope guardrails.


## Stage 6 assignment PPO

Stage 6 learns only parcel assignment decisions at `PARCEL_ARRIVAL` events. Its
categorical action space is `1 + 2H` (`TD`, `TBD_h`, and `TLD_h`), and infeasible
logits are masked before sampling. Bus charging is never learned in this stage;
choose `no_charge`, `uniform_30`, or `battery_threshold` in the configuration.

The clipped objective uses `ratio = exp(new_log_prob - old_log_prob)`, normalized
GAE advantages, clipped policy loss, MSE value loss, entropy regularization, and
gradient clipping. Train, evaluate, or run the temporary-artifact smoke test with:

```bash
python -m experiments.train_assignment_ppo --config configs/train_assignment_ppo.yaml
python -m experiments.evaluate_assignment_ppo --config configs/train_assignment_ppo.yaml --checkpoint results/checkpoints/assignment_ppo.pt
python -m experiments.smoke_test_assignment_ppo
```

With `rlaif.enabled: false`, total reward is the finite event-to-event environment
assignment reward and no `reward_model.pt` is required. With `rlaif.enabled: true`,
the configured Stage 5 checkpoint is mandatory and its saved feature and reward
normalization statistics are applied. Missing or invalid checkpoints fail clearly;
rule-based, reason-text, and fabricated RLAIF rewards are prohibited. PyTorch is
required for model training, updates, checkpoint round trips, and evaluation. A
skipped PyTorch smoke test is only a Code Gate result, not experimental validation.

## Stage 7 asynchronous MAPPO

Stage 7 follows the simulator's event queue rather than inventing simultaneous
multi-agent timesteps. Exactly one decentralized actor is active at a decision:
the assignment actor at `PARCEL_ARRIVAL`, or the bus actor at an integrated-station
`BUS_ARRIVAL`. Truck routing and drone dispatch remain deterministic environment
modules, and no transition is fabricated for the inactive actor.

The assignment policy is a masked categorical distribution with `1 + 2H` actions.
The bus policy is a separate masked categorical distribution over
`[0, 15, 30, 45, 60, 75, 90, 105, 120]` charging seconds. Both actors train against
one shared centralized critic using `env.get_global_state()`. The buffer records
one row per real decision event and computes normalized GAE over that asynchronous
event stream.

```bash
python -m experiments.train_mappo_async --config configs/train_mappo_async.yaml
python -m experiments.evaluate_mappo_async --config configs/train_mappo_async.yaml --checkpoint results/checkpoints/mappo_async.pt
python -m experiments.smoke_test_mappo_async
```

`rlaif.enabled: false` is the dependency-light smoke mode and never requires
`reward_model.pt`. `rlaif.enabled: true` strictly requires a valid trained Stage 5
checkpoint; learned reward is added only to assignment transitions. Bus rewards
never include RLAIF. Rule scores, evaluator text, and fabricated fallback rewards
are prohibited. PyTorch is required for networks, optimization, and checkpoint
round trips, so a clean skip without PyTorch is not experimental validation.
Final RLAIF-enabled MAPPO experiments remain blocked until the Stage 5 Runtime
Gate passes in a PyTorch environment. Stage 8 experiments are not part of Stage 7.

## Stage 8 experiment framework

Stage 8 adds dependency-light baselines, fair seed-controlled evaluation, benchmark/ablation/sensitivity runners, and CSV/JSON aggregation. Validate the wiring without trained models or large experiments:

```bash
python -m experiments.smoke_test_experiments
```

Run configured workflows later in the proper experiment environment:

```bash
python -m experiments.run_benchmark --config configs/experiments.yaml
python -m experiments.run_ablation --config configs/ablation.yaml
python -m experiments.run_sensitivity --config configs/sensitivity.yaml
python -m experiments.aggregate_results --input results/raw --output results/summary
```

Outputs live under the configured `results/` directory and are ignored by Git. Missing learned checkpoints and PyTorch dependencies are explicit skips; no heuristic substitution occurs. `rlaif_enabled=false` does not require `reward_model.pt`, while enabled RLAIF requires a valid trained Stage 5 checkpoint. Baseline heuristics are comparisons only and must never become RLAIF labels. Current smoke outputs are code-validation artifacts, not final experimental results; final experiments remain deferred pending the Stage 5 Runtime Gate and trained Stage 6/7 policies. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Experiment-ready paper alignment

For the paper “Preference-Aligned Multi-Agent Reinforcement Learning for Dynamic Truck--Electric-Bus--Drone Parcel Delivery”, use `configs/train_mappo_async.yaml` as the official MAPPO config and follow `docs/EXPERIMENT_GUIDE.md`. The old `configs/train_mappo.yaml` is deprecated.

Current RLAIF support is assignment-agent-only: AI preferences train a reward model for assignment choices, while truck, bus, and station rewards remain pure environment rewards. API calls are used only offline to generate preference data via `RLAIF_API_KEY`, `RLAIF_API_BASE_URL`, and `RLAIF_MODEL_NAME`; MAPPO training does not call an API in real time. The station agent action space is `dispatch_drone` or `idle`; battery recharging is an environment dynamic after drone dispatch/return, not a learned station action.

## Phase 0 paper contract and parameter schema

The formal Phase 0 paper-code contract is documented in `docs/paper_code_alignment/final_dynamic_contract.md`. The repository now includes schema-level entities, event types, and validators for the target dynamic multi-agent architecture, but later-phase capabilities remain **specified** rather than implemented: truck batching, physical-bus circulation, passenger dynamics, station battery decisions, and multi-agent RLAIF.

Formal paper configurations live in `configs/paper/`: `base_small.yaml`, `base_medium.yaml`, `base_large.yaml`, `train_mappo_env.yaml`, and `train_mappo_rlaif.yaml`. The legacy 20-episode MAPPO config is smoke-only and should not be used as the formal paper training configuration.
