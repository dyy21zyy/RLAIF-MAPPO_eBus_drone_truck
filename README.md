# RLAIF-MAPPO for Dynamic Multimodal Parcel Delivery

This repository is being developed in explicit stages for parcel assignment and
scheduling across trucks, electric buses, integrated stations, and drones.

## Current status: Stage 5 Code Gate complete; Runtime Gate deferred

Stage 2 provides a reproducible Shanghai instance data pipeline with two road
network modes:

- **Full mode:** attempts an OpenStreetMap drivable-network download through
  `osmnx` when that optional package is installed, and automatically falls back
  if the download fails.
- **Fallback mode:** uses a deterministic 25-node directed grid, a generated
  20-stop corridor route, a synthetic timetable and parcels, and requires no
  internet access.

Stage 3 adds the deterministic event-driven assignment and electric-bus charging
MDP. Stage 4 adds assignment-state collection, objective candidate features,
versioned pairwise AI prompts, and offline/API/replay preference validation.
Stage 5 adds a learned assignment reward model trained only from approved
pairwise labels. Its dependency-light Code Gate passes in the current environment;
the Runtime Gate is deferred to an environment with PyTorch. PPO and MAPPO remain
intentionally **not implemented**.

## Repository layout

```text
configs/        Scenario and future training configuration templates
data/           Raw inputs and ignored generated instances
data_pipeline/  Stage 2 road, bus, facility, parcel, matrix, and instance builders
checkpoints/    Future model artifacts
docs/           Design, guardrail, and experiment documentation
envs/           Stage 3 event-driven assignment and bus environment
experiments/    Offline gates and Stage 4 data-workflow CLIs
logs/           Runtime logs
models/         Future model placeholder
outputs/        Future experiment outputs
rlaif/          Stage 4 preference workflow and Stage 5 reward-model training
tests/          Stage 1 through Stage 5 regression tests
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
- Stage 3: event-driven MDP environment (complete).
- Stage 4: RLAIF state/prompt collection and AI-label interface (complete).
- Stage 5: Code Gate complete; PyTorch Runtime Gate deferred.
- Stage 6: assignment-only PPO code and dependency-light Code Gate implemented.
  The bus uses a fixed baseline; Stage 6 contains no MAPPO or centralized critic.
- Stage 7: not implemented. Final RLAIF-enabled experiments remain blocked until
  `reward_model.pt` has passed the deferred Stage 5 Runtime Gate.

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
