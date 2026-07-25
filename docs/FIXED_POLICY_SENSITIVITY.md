# Fixed-policy robustness sensitivity

This workflow evaluates the **same three trained full-RLAIF asynchronous MAPPO
policies** as one operating condition changes. It is not retrained-policy
sensitivity: actors, critics, four reward models, RLAIF weights/clipping,
reference scales, architecture, and deterministic masked actions remain fixed.

The one-factor families are passenger intensity (`0.75, 1.00, 1.25, 1.50`),
parcel count (`45, 60, 75, 90`), and station power kW (`880, 1100, 1320`). Each
level uses seeds 310000–310099 and a paired index. Sensitivity banks explicitly
are not the final paper test bank. Reward models audit selected transitions and
never select actions. Missing lineage, features, required instrumented metrics,
termination, hashes, or any fallback fails closed. Passenger averages use the
explicit boarding count, never parcels.

Aggregation averages three policy seeds per value and paired index before it
summarizes 100 scenarios. It reports seeded bootstrap intervals and paired
Wilcoxon tests, rank-biserial effects, and Holm correction. All-zero differences
are explicitly p=1/effect=0.

## AutoDL formal commands

```bash
git fetch origin && git switch main && git pull --ff-only
for s in 1 2 3; do test -f results/formal/mappo_rlaif_all/seed_$s/mappo_rlaif_all_seed_$s.pt; done
for a in assignment truck bus station; do test -f results/formal/reward_models/reward_$a.pt; done
test -f results/formal/reward_scales/final_reward_reference_scales.json
python -m experiments.prepare_fixed_policy_sensitivity_inputs --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed
python -m experiments.run_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed --validate-only
python -m experiments.run_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed --family passenger
python -m experiments.run_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed --family parcel
python -m experiments.run_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed --family power
python -m experiments.aggregate_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed
python -m experiments.aggregate_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/formal/sensitivity_fixed --validate-only
```

Preparation `--resume` reuses only fully validated count/seed/index/config/hash
lineage. Evaluation `--resume` skips only an exact successful identity. `--force`
is restricted to the sensitivity root and cannot remove formal scenario banks,
policies, reward models, or scales.

## Thirty-episode diagnostic

Diagnostic evidence is separate and publication-ineligible. Five scenarios for
seed 1 at six extremes produce exactly 30 episodes:

```bash
python -m experiments.prepare_fixed_policy_sensitivity_inputs --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/diagnostic/sensitivity_fixed --diagnostic-scenario-count 5 --force
for item in 'passenger .75' 'passenger 1.50' 'parcel 45' 'parcel 90' 'power 880' 'power 1320'; do set -- $item; python -m experiments.run_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/diagnostic/sensitivity_fixed --family "$1" --value "$2" --policy-seed 1 --scenario-limit 5; done
python -m experiments.run_fixed_policy_sensitivity --config configs/paper/fixed_policy_sensitivity.yaml --output-root results/diagnostic/sensitivity_fixed --validate-only --policy-seed 1
```

Real checkpoints and generated banks are ignored runtime artifacts. Without
them, only fixture tests and artifact-presence validation can run; diagnostic
output is not publication-ready and no formal result should be claimed.
