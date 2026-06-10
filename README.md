# RLAIF-MAPPO for Dynamic Multimodal Parcel Delivery

This repository is being developed in explicit stages for parcel assignment and
scheduling across trucks, electric buses, integrated stations, and drones.

## Current status: Stage 3

Stage 2 provides a reproducible Shanghai instance data pipeline with two road
network modes:

- **Full mode:** attempts an OpenStreetMap drivable-network download through
  `osmnx` when that optional package is installed, and automatically falls back
  if the download fails.
- **Fallback mode:** uses a deterministic 25-node directed grid, a generated
  20-stop corridor route, a synthetic timetable and parcels, and requires no
  internet access.

Stage 3 adds the deterministic event-driven assignment and electric-bus charging
MDP. It consumes Stage 2 manifests, exposes stable action masks and feature
schemas, tracks delayed parcel costs and physical resources, and retains an
offline end-to-end smoke gate. PPO, MAPPO, preference generation, reward
modeling, and RLAIF are intentionally **not implemented**.

## Repository layout

```text
configs/        Scenario and future training configuration templates
data/           Raw inputs and ignored generated instances
data_pipeline/  Stage 2 road, bus, facility, parcel, matrix, and instance builders
checkpoints/    Future model artifacts
docs/           Design, guardrail, and experiment documentation
envs/           Stage 3 event-driven assignment and bus environment
experiments/    Offline stage-gate smoke tests
logs/           Runtime logs
models/         Future model placeholder
outputs/        Future experiment outputs
tests/          Stage 1 through Stage 3 regression tests
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
python -m experiments.smoke_test_environment --config configs/shanghai_small.yaml
```

See [docs/MDP_SPECIFICATION.md](docs/MDP_SPECIFICATION.md) for event, action,
transition, reward, and termination semantics.

## Verification

```bash
python -m experiments.smoke_test_project --config configs/shanghai_small.yaml
python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback
python -m experiments.smoke_test_environment --config configs/shanghai_small.yaml
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
- Later stages: PPO/MAPPO and RLAIF components (not started).

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the staged workflow and
[docs/PITFALLS.md](docs/PITFALLS.md) for scope guardrails.
