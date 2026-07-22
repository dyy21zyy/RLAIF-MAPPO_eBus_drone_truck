# Fix 8b Real Readiness Pilot Findings

| file | function | current synthetic behavior | required real behavior | failing test |
|---|---|---|---|---|
| `experiments/run_readiness_pilot.py` | `generate` | Manually wrote bus, passenger, station-power, event, and reward-ledger rows. | Export traces from `DynamicDeliveryEnv` runtime collectors. | `tests/test_real_readiness_pilot_no_synthetic_data.py` |
| `experiments/run_readiness_pilot.py` | `mappo_report` | Used random tensors for observations/candidates and a diagnostic fake checkpoint byte string. | Collect `AsyncTransition` objects through `collect_episode`, call `update_mappo`, and save a production MAPPO checkpoint. | `tests/test_real_readiness_pilot_actor_updates.py` |
| `experiments/run_readiness_pilot.py` | `classify` | Could report formal completion/ready from diagnostic values. | Report `ENV_MAPPO_READY_RLAIF_BLOCKED` when environment/MAPPO gates pass and formal reward checkpoints are absent. | `tests/test_real_readiness_status.py` |
| `envs/delivery_env.py` | runtime event handling | Bus trace existed, but event/truck/parcel trace exports were not available to the pilot. | Instrument real processed events and decision actions, then export runtime trace rows. | `tests/test_real_readiness_pilot_event_coverage.py` |
| `training/mappo_trainer.py` | checkpoint path | Production writer existed but pilot bypassed it. | Pilot must call `save_checkpoint` and reload fresh modules with `load_checkpoint`. | `tests/test_real_readiness_pilot_checkpoint_roundtrip.py` |
