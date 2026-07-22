# Fix 8a Environment Semantics Findings

Starting main SHA: `17ab5af22711c5bcd1ee8b75a50f496d48e7835d`

| Defect | file | class/function | current behavior | required behavior | failing test |
|---|---|---|---|---|---|
| Parcel-dependent bus timetable shift | `envs/delivery_env.py` | `DynamicDeliveryEnv.reset` | Trip start events at or before first parcel release are shifted to `first_parcel_release + EPSILON`. | Schedule every trip at its timetable departure when within horizon. | `tests/test_bus_timetable_independent_of_parcels.py` |
| Conflicting event priority definition | `envs/delivery_env.py` | `EVENT_PRIORITY` | Priority table is local to the environment module. | Use one canonical priority table. | `tests/test_event_priority_same_time_determinism.py` |
| Constant candidate base load | `envs/action_generators/station_actions.py` | `_station_base_load`, `projected_load` | Candidate projected load reads station/config constant load. | Use `env.station_base_load_profile.load_at(station_id, now)`. | `tests/test_station_candidate_uses_time_varying_base_load.py` |
| Hard-coded drone feasibility constants | `envs/action_generators/station_actions.py` | `_dispatch_pattern` | Payload/radius/max round trip use module constants `5.0/8.0/120.0`. | Resolve from runtime/config with formal fail-closed behavior. | `tests/test_drone_sensitivity_changes_feasibility.py` |
| Hard-coded charging/station constants | `envs/action_generators/station_actions.py`, `envs/delivery_env.py` | candidate generation and station construction | Charging slots/power/duration and capacity have fallback constants in runtime paths. | Prefer entity/config values; allow defaults only for explicit smoke/diagnostic fallback. | `tests/test_station_operational_parameters_from_config.py` |
| Formal fallback | `configs/paper/train_mappo_env.yaml`, `configs/paper/train_mappo_rlaif.yaml` | config | `env.fallback: true`. | Formal configs must set `run_classification: formal` and `env.fallback: false`. | `tests/test_formal_config_rejects_fallback.py` |
| Missing reward-scale hash | `configs/paper/*.yaml` | reward config | `scale_artifact_hash: null`. | Non-placeholder hash required when reference scales are enabled. | `tests/test_formal_config_reward_scale_gate.py` |
| Truck-cost weight with zero coefficients | `training/config_resolver.py` | formal validation | No consistency check between reward weight and truck coefficients. | Reject positive truck-cost reward when all truck cost coefficients are zero. | `tests/test_formal_config_truck_cost_gate.py` |
