# Step 4 RLAIF Runtime Findings

Starting main SHA: `c4816b44d3f165aa06dcc25734e68485d3350024`.

| file | class or function | current behavior | required behavior | failing test |
|---|---|---|---|---|
| `training/reward_model_wrapper.py` | `RewardModelWrapper.score` | Legacy runtime fields such as `state_schema_version`, `state_feature_mean`, `candidate_feature_mean`, `reward_mean`, `reward_std`, and `model_config` were used for runtime scoring. | Runtime must consume Phase 5B canonical fields and strict loader validation. | `tests/test_runtime_reward_checkpoint_canonical_loader.py` |
| `training/reward_model_wrapper.py` | `RewardModelWrapper.score` | Event embedding rows were derived from `compatible_event_types.index(event_type)`. | Use canonical global `EVENT_NAME_TO_ID[event_type]`. | `tests/test_runtime_reward_event_id_mapping.py` |
| `rlaif/reward_registry.py` | `RewardRegistry._load` | Registry constructed the legacy wrapper for all checkpoints. | Load `RuntimeAgentRewardModel` for canonical checkpoints. | `tests/test_reward_registry_full_scope.py` |
| `rlaif/reward_registry.py` | `RewardRegistry.score_transition` | Runtime score was already normalized/clipped, then normalized/clipped again with YAML defaults. | Keep raw, normalized, clipped, and weighted stages separate using checkpoint output stats. | `tests/test_reward_contribution_stage_separation.py` |
| `rlaif/reward_registry.py` | `RewardRegistry.__init__` | Formal-mode fallback validation was implicit and incomplete. | Formal mode rejects fallback, partial full scope, missing checkpoints, and bad hashes. | `tests/test_reward_registry_formal_fail_closed.py` |
| `training/mappo_trainer.py` | rollout transition recording | Existing code had risk of storing one learned scalar in multiple learned reward fields. | Store raw, normalized, clipped, weighted, and total rewards from `RewardContribution`. | `tests/test_mappo_transition_reward_decomposition.py` |
| `training/mappo_trainer.py` / CLI | rollout formal mode | Formal mode could be lost or hard-coded false in rollout collection paths. | Propagate `run_classification=formal` to registry and checkpoint loader. | `tests/test_mappo_formal_mode_propagation.py` |
| `training/mappo_checkpoint.py` / `training/mappo_trainer.py` | policy checkpoint save/load | Policy lineage preserved partial reward metadata. | Store exact reward checkpoint paths, hashes, schema versions, validation statuses, lambdas, clips, and scope. | `tests/test_mappo_rlaif_policy_checkpoint_lineage.py` |
