# Fix Phase 4 findings

Starting main SHA: `2ad7e1251bea2273d0dfd9c02d7f208ea503dc5b`.

| Defect | file | class/function | current behavior confirmed | required behavior | failing test |
| --- | --- | --- | --- | --- | --- |
| Helper-only event embedding | `training/mappo_networks.py` | `CandidateScoringActor` | scorer input concatenated observation and candidate features only | concatenate observation, learned event embedding, and candidate features | `tests/test_event_embedding_enters_actor.py` |
| Actor logits event-insensitive | `training/mappo_networks.py` | `forward`/`act` | no explicit event ID argument | logits can differ for fixed non-event inputs | `tests/test_event_embedding_changes_logits.py` |
| Buffer omitted event ID | `training/mappo_buffer.py` | `AsyncTransition` | stored event name but not canonical ID | store canonical name and matching ID | `tests/test_mappo_buffer_event_ids.py` |
| Legacy bus names propagated | `training/mappo_async.py`, `envs/delivery_env.py` | trainer boundary | `BUS_DEPARTURE`/`BUS_ARRIVAL` appeared in rollout probing | normalize at boundary to canonical names | `tests/test_decision_event_schema.py` |
| Bus probe incomplete | `training/mappo_trainer.py` | `_collect_actor_specs` | stopped after one legacy bus arrival | require all agent-event coverage including both bus events | `tests/test_mappo_actor_spec_event_coverage.py` |
| Checkpoint identity ambiguous | `training/mappo_trainer.py` | `save_checkpoint` | single env-reward algorithm string | explicit env/assignment/all RLAIF identities | `tests/test_mappo_checkpoint_algorithm_identity.py` |
| Reward double-scaling risk | `rlaif/reward_registry.py`, `training/mappo_trainer.py` | `total_reward`, `collect_episode` | registry returned unstructured learned reward | structured weighted contribution added once | `tests/test_rlaif_no_global_double_scaling.py` |
| Per-agent provenance lost | `rlaif/reward_registry.py` | reward totals | totals could be aggregated without agent provenance | per-agent raw and weighted reconciliation | `tests/test_rlaif_training_log_reconciliation.py` |
| Fallback too permissive | `rlaif/reward_registry.py` | constructor/score | fallback flags could be silent | formal fail-closed; smoke fallback explicit | `tests/test_rlaif_formal_fail_closed.py`, `tests/test_rlaif_smoke_fallback.py` |
