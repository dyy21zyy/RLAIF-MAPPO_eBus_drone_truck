# Fix 8L Formal Go-Live Findings

Starting local main SHA: `a0a11b4966671c4bd23564cc0c20e7cf79ff2a43`.

Remote reproduction note: this workspace had no configured `origin`, so `git fetch origin` failed with `fatal: 'origin' does not appear to be a git repository` before branch creation. The repair branch was created from the checked-out local `main` snapshot.

## Reproduced blockers

A. Delivery horizon schema mismatch:

```bash
rg -n "delivery_horizon_min|delivery_evaluation_horizon_min" configs experiments training data_pipeline envs tests -S
```

The canonical paper configs define `time.delivery_evaluation_horizon_min`, while `DynamicDeliveryEnv.reset()` read `config["bus"]["delivery_horizon_min"]` directly. This could fail for formal configs that had not passed through the legacy instance-builder alias path.

B. Reward-scale estimation resumability:

```bash
python -m experiments.estimate_reward_reference_scales --help
```

The CLI exposed `--resume`, but the implementation did not persist per-scenario progress before aggregation; an interrupted formal train-bank run had to restart completed rollouts.

C. Missing formal assignment reward-model checkpoint:

```bash
find results/formal/reward_models -maxdepth 1 -type f -name 'reward_assignment.pt' -print
```

No formal checkpoint was present in the runtime artifact directory in this workspace.

D. False-success formal training skip:

```bash
rg -n "SKIP:.*PyTorch|return 0" experiments/train_assignment_ppo.py experiments/train_mappo_async.py
```

Both training entry points returned success when PyTorch was unavailable, regardless of formal run classification.
