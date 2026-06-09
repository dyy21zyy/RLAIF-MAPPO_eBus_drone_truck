# Experiment Catalog

> Stage 1 supplies planning templates only; no training experiment is implemented.

## Naming convention

Use `<stage>_<purpose>_<city>_<seed>_<timestamp>` for artifact directories.
Store the resolved configuration and a metadata manifest with every run.

## Planned experiment families

| Family | Config | Status | Primary purpose |
| --- | --- | --- | --- |
| Project smoke | `configs/shanghai_small.yaml` | Implemented | Validate Stage 1 foundation |
| Data pipeline smoke | `configs/shanghai_small.yaml` | Stage 2 planned | Validate fallback instance |
| Environment smoke | `configs/shanghai_small.yaml` | Stage 3 planned | Complete one valid episode |
| Reward model | `configs/train_reward_model.yaml` | Future template | Fit preference reward |
| Assignment PPO | `configs/train_assignment_ppo.yaml` | Future template | Assignment-only baseline |
| MAPPO | `configs/train_mappo.yaml` | Future template | Cooperative assignment/bus policy |

## Minimum reporting checklist

- Commit and clean/dirty status
- Fully resolved configuration
- Seed set
- Dataset/instance identifier
- Exact command
- Runtime and hardware
- Success/failure status
- Metrics with definitions
- Artifact paths
- Known limitations
