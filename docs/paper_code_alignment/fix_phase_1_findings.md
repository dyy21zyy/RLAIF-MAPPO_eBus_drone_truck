# Fix Phase 1 findings

Starting main SHA: `e368873cd1c6c2d93df7cf63b1453ad4e775aaf8`.

## Confirmed root causes and proof tests

| Issue | File | Function/config path | Current behavior | Required behavior | Test proving fix |
| --- | --- | --- | --- | --- | --- |
| Incomplete formal training config | `configs/paper/train_mappo_env.yaml`, `configs/paper/train_mappo_rlaif.yaml` | `env`, `training.seed`, `output.*` | Formal configs mixed paper hyperparameters with missing runtime trainer fields. | Resolve to a complete one-seed runtime config before training. | `tests/test_formal_training_config_runtime.py` |
| Mixed hyperparameter aliases | `training/mappo_trainer.py`, `training/ppo_trainer.py` | PPO update loss coefficients | Runtime used `ent_coef`/`vf_coef` while formal configs used `entropy_coef`/`value_coef`. | Resolver canonicalizes aliases and rejects conflicts; trainers accept canonical keys. | `tests/test_training_config_key_normalization.py` |
| All-zero formal reward | `data_pipeline/build_instance.py`, `configs/paper/base_*.yaml` | `normalize_dynamic_config`, `reward` | Missing reward blocks could be defaulted to all zeros. | Formal configs carry explicit nonzero weights; validation rejects missing/all-zero/invalid weights. | `tests/test_formal_reward_nonzero.py` |
| Reward scales not applied | `envs/reward_ledger.py`, `envs/delivery_env.py` | `RewardLedger.add_cost`, `_charge_cost` | Weighted amount used raw cost times weight. | Use normalized amount `raw / scale` and weighted amount `normalized * weight`. | `tests/test_reward_reference_scale_application.py` |
| Status mismatch | `training/mappo_trainer.py`, `training/ppo_trainer.py`, `evaluation/metrics.py`, `envs/delivery_env.py` | episode summaries and metrics | Some summaries checked lowercase `delivered`; environment uses uppercase `DELIVERED`. | Shared status normalization helper. | `tests/test_parcel_status_consistency.py` |
| Missing urgent runtime field | `data_pipeline/generate_parcels.py`, `envs/delivery_env.py`, `evaluation/metrics.py` | parcel generation/loading/formal metrics | Runtime parcels did not reliably expose `is_urgent`; metrics could fall back to priority. | Load `deadline_class` and define `is_urgent = deadline_class == "tight"`. | `tests/test_runtime_urgent_class.py` |
| Silent formal metric zeros | `evaluation/metrics.py` | `collect_formal_metrics` | Required metrics used missing-attribute fallbacks that could look like real zeros. | Strict `collect_formal_runtime_metrics` raises `FormalMetricError`; legacy wrapper is separated. | `tests/test_formal_metric_runtime_mapping.py`, `tests/test_formal_metric_no_silent_zero.py` |
| Relocation/layover mismatch | `data_pipeline/build_instance.py`, `data_pipeline/build_bus_circulation.py` | `bus_schedule.relocation_time_min`, `bus_schedule.minimum_layover_min` | Legacy defaults `5`/`2` could be used before paper aliases. | Map paper aliases first and reject conflicting aliases. | `tests/test_relocation_layover_config_mapping.py` |
| Output paths/three seeds | `experiments/train_policy_matrix.py`, `training/config_resolver.py` | `training.training_seeds`, `output.*_template` | Three-seed formal runs were not resolved to independent concrete output paths. | Resolve one scalar seed at a time and reject duplicate paths. | `tests/test_formal_training_seed_matrix.py` |
| Smoke vs formal settings | `configs/shanghai_small.yaml`, `configs/paper/base_*.yaml` | `reward.apply_reference_scales` | Smoke and formal reward behavior were not explicitly separated. | Formal configs enable scale artifacts; smoke configs can keep simplified unscaled rewards. | `tests/test_formal_reward_nonzero.py` plus smoke commands |

## Deferred to Fix Phase 2

The stop-by-stop bus event chain was intentionally not modified in this phase.
