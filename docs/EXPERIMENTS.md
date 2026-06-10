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

## Stage 6 assignment PPO commands

```bash
# Dependency-light code smoke test; uses rlaif.enabled=false and a temporary directory.
python -m experiments.smoke_test_assignment_ppo

# Training (requires PyTorch).
python -m experiments.train_assignment_ppo --config configs/train_assignment_ppo.yaml

# Deterministic evaluation with the same fixed bus baseline (requires PyTorch).
python -m experiments.evaluate_assignment_ppo --config configs/train_assignment_ppo.yaml --checkpoint results/checkpoints/assignment_ppo.pt
```

The actor and critic are separate `[256, 256]` ReLU MLPs. The actor samples a
masked categorical distribution over `1 + 2H` assignment actions; the critic is a
local state-value model, not a centralized critic. PPO uses normalized GAE,
clipped likelihood ratios, MSE value loss, entropy regularization, and gradient
clipping. Rollout storage contains assignment transitions only.

The default config disables RLAIF, so smoke testing does not require Stage 5
runtime output. Enabling RLAIF requires a valid trained checkpoint and never
activates a rule fallback. Training CSVs, evaluation JSON, and checkpoints under
`results/` are runtime artifacts and are not committed. These commands are code
and runtime checks, not Stage 8 experiments or final RLAIF-enabled validation.
