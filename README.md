# RLAIF-MAPPO for Dynamic Multimodal Parcel Delivery

This repository is being developed in explicit stages for parcel assignment and
scheduling across trucks, electric buses, integrated stations, and drones.

## Current status: Stage 1

Stage 1 provides only the stable project foundation:

- YAML configuration loading;
- console/file logger setup;
- deterministic seed control for Python and optional NumPy/PyTorch backends;
- standard runtime-directory creation;
- importable placeholders for later pipeline, environment, model, and training code;
- an offline project smoke test;
- workflow, experiment, MDP, feature, and RLAIF documentation templates.

The simulator, Shanghai data pipeline, PPO/MAPPO, and RLAIF are intentionally
**not implemented** in this stage.

## Repository layout

```text
configs/        Concrete experiment and training configuration templates
data/           Raw and processed data roots (empty in Stage 1)
data_pipeline/  Stage 2 placeholder
checkpoints/    Future model artifacts
docs/           Design and experiment documentation
envs/           Stage 3 placeholder
experiments/    Executable smoke tests and future experiments
logs/           Runtime logs
models/         Future model placeholder
outputs/        Future experiment outputs
tests/          Future unit tests
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

The Stage 1 smoke test does not access the internet. Installing dependencies may
be performed ahead of time in an offline environment with a prepared wheel cache.

## Stage 1 smoke test

From the repository root, run:

```bash
python -m experiments.smoke_test_project --config configs/shanghai_small.yaml
```

The command checks configuration loading, runtime folders, file logging,
deterministic seeds, and imports of all future-stage placeholders.

## Configuration

`configs/shanghai_small.yaml` is the concrete MVP scenario configuration. The
other files in `configs/` are templates that reserve stable names for later
reward-model, assignment-PPO, and MAPPO work; they are not executable trainers.

## Development boundaries

- Stage 1: foundation and documentation (current).
- Stage 2: offline-capable Shanghai instance data pipeline.
- Stage 3: event-driven MDP environment.
- Later stages: PPO/MAPPO and RLAIF components.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the staged workflow and
[docs/PITFALLS.md](docs/PITFALLS.md) for scope guardrails.
