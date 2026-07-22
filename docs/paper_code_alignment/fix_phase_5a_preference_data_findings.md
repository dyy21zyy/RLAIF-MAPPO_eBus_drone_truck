# Fix Phase 5A Preference Data Findings

| file | function | current behavior | required behavior | failing test |
| --- | --- | --- | --- | --- |
| `rlaif/grouped_split.py` | `grouped_split` | grouped only by state and accepted pair-level defaults via ratios. | Support state/episode/scenario groups, validate fractions, and prove disjoint groups. | `tests/test_reward_model_grouped_split_no_leakage.py` |
| `rlaif/preference_schema_v2.py` | `validate_preference_state` | validates preference states but not labeled A/B preference records with provenance. | Canonical v3 preference records with original A/B order, display order, original outcome, and schema-version checks. | `tests/test_preference_schema_v3.py` |
| `rlaif/preference_schema_v2.py` | `validate_agent_event` | has event compatibility, but no label-source validation or original winner semantics. | Reject incompatible labels and preserve winner in original A/B order. | `tests/test_preference_label_source_validation.py` |
| `rlaif/reward_model_dataset.py` | dataset builder | no complete canonical pair dataset implementation was present. | Build strict four-agent `RewardPairDataset` and exclude ties/abstentions/invalid/unresolved. | `tests/test_reward_model_dataset_real.py` |
| `rlaif/reward_model_dataset.py` | duplicate handling | contradictory duplicates were not conservatively resolved in the canonical dataset. | Deduplicate identical pair/winner and exclude contradictory duplicate pairs. | `tests/test_reward_model_duplicate_contradiction_handling.py` |
| `rlaif/train_reward_model.py` | `compute_feature_statistics` | legacy assignment path can normalize after ad hoc split semantics. | Phase 5A helper must fit frozen state/candidate normalizers using training split only. | `tests/test_reward_model_normalization_train_only.py` |
| `experiments/summarize_preference_dataset.py` | CLI | existing summary did not emit complete v3 integrity gates. | Emit integrity JSON with counts, leakage status, and bus coverage checks. | `tests/test_reward_model_dataset_integrity.py` |
| `configs/paper/train_reward_*.yaml` | config | formal configs used v2 fields without run classification or scenario grouping. | Mark formal mode and scenario-group split fractions. | focused config verification |
