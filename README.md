# RLAIF-MAPPO for Dynamic Multimodal Parcel Delivery

This repository is being developed in explicit stages for parcel assignment and
scheduling across trucks, electric buses, integrated stations, and drones.

## Current status: Stage 2

Stage 2 provides a reproducible Shanghai instance data pipeline with two road
network modes:

- **Full mode:** attempts an OpenStreetMap drivable-network download through
  `osmnx` when that optional package is installed, and automatically falls back
  if the download fails.
- **Fallback mode:** uses a deterministic 25-node directed grid, a generated
  20-stop corridor route, a synthetic timetable and parcels, and requires no
  internet access.

Stage 1 configuration, logging, seeding, and project-foundation utilities remain
available. The event-driven simulator, PPO, MAPPO, preference generation,
reward modeling, and RLAIF are intentionally **not implemented**.

## Repository layout

```text
configs/        Scenario and future training configuration templates
data/           Raw inputs and ignored generated instances
data_pipeline/  Stage 2 road, bus, facility, parcel, matrix, and instance builders
checkpoints/    Future model artifacts
docs/           Design, guardrail, and experiment documentation
envs/           Stage 3 placeholder only
experiments/    Stage 1 and Stage 2 smoke tests
logs/           Runtime logs
models/         Future model placeholder
outputs/        Future experiment outputs
tests/          Stage 1 and Stage 2 regression tests
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

## Verification

```bash
python -m experiments.smoke_test_project --config configs/shanghai_small.yaml
python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback
python -m pytest -q
python -m compileall -q .
git diff --check
```

The Stage 2 smoke test forces fallback mode and validates the complete artifact
set, route/station/parcel invariants, matrix shapes, and both manifests without
making any network request.

## Development boundaries

- Stage 1: foundation and documentation (complete).
- Stage 2: offline-capable Shanghai instance data pipeline (complete).
- Stage 3: event-driven MDP environment (not started).
- Later stages: PPO/MAPPO and RLAIF components (not started).

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the staged workflow and
[docs/PITFALLS.md](docs/PITFALLS.md) for scope guardrails.
