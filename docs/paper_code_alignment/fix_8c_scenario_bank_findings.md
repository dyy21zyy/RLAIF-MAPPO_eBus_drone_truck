# Fix 8c scenario-bank findings

Starting SHA: `8cc8cefb93cb8145f6e0283a8b80f05c5d00b9e1`.

| Defect | File/function | Current behavior | Required behavior | Failing test |
|---|---|---|---|---|
| Seed tuple not used | `experiments/build_scenario_bank.py::build_bank` | Builds seed dict but calls `build_instance(... fallback=True ...)` without seed overrides. | Pass canonical seed tuple into the single instance-generation path. | `tests/test_scenario_seed_overrides_reach_generators.py` |
| Same artifacts under different IDs | `experiments/build_scenario_bank.py::build_bank` | Scenario ID changes while generation seeds remain base-config seeds. | Generated dynamic artifacts must be controlled by scenario tuple. | `tests/test_scenario_different_seeds_change_dynamic_artifacts.py` |
| Incomplete freeze | `evaluation/scenario_bank.py::freeze_scenario` | Copies only `instance.json`. | Copy every artifact referenced by `instance.json` and fail if missing. | `tests/test_frozen_scenario_is_self_contained.py` |
| Stale paths | `evaluation/scenario_bank.py::freeze_scenario` | Leaves source output directory. | Rewrite frozen instance paths to relative names and provenance separately. | `tests/test_frozen_scenario_rewrites_paths.py` |
| Volatile hashes | `evaluation/scenario_bank.py::write_bank_manifest` | Hashes include generated manifest details without canonical content separation. | Exclude volatile fields from content and bank hashes. | `tests/test_scenario_same_seed_reproducibility.py` |
| Single training env | `training/mappo_trainer.py::train_mappo_async` | Creates one env and resets it. | Create fresh env per sampled frozen scenario. | `tests/test_mappo_training_uses_multiple_scenarios.py` |
| Missing episode provenance | `training/mappo_trainer.py::train_mappo_async` | CSV has only episode metrics. | Log scenario ID, split, content hash, instance hash, bank hash, position, cycle. | `tests/test_mappo_training_scenario_log_provenance.py` |
| Missing checkpoint lineage | `training/mappo_trainer.py::save_checkpoint` | Does not include concrete bank path/hash/scenarios. | Store scenario-bank lineage in checkpoints. | `tests/test_mappo_checkpoint_scenario_lineage.py` |
| Formal bank not required | `training/mappo_trainer.py::train_mappo_async` | Can train formal from `env.config_path`. | Reject formal training without `env.scenario_bank_manifest`. | `tests/test_mappo_training_uses_multiple_scenarios.py` |
