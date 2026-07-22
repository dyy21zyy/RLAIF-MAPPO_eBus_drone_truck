# Fix Phase 3 Passenger and Station-Power Findings

## Confirmed defects

| Issue | File | Function | Current behavior | Required behavior | Failing test |
|---|---|---|---|---|---|
| Homogeneous passenger process | `envs/dynamics/passenger_dynamics.py` | `generate_arrival_events` | Samples one stop rate, one full-horizon Poisson count, and uniform arrival times over the horizon. | Generate per temporal block with rate `baseline × intensity × multiplier`. | `tests/test_time_dependent_passenger_process.py` |
| Effective rate clipped | `envs/dynamics/passenger_dynamics.py` | `sample_stop_rates` | Clips intensity-scaled rate to `[0.05, 0.60]`. | Clip only baseline rate; effective may exceed `0.60`. | `tests/test_passenger_intensity_scaling.py` |
| Queue delay tied to stop visit | `envs/dynamics/passenger_dynamics.py` | `PassengerArrivalIndex.apply_until` | Integrates one stop when visited. | Central runtime integrates all queues whenever time advances and at horizon. | `tests/test_passenger_horizon_waiting_integration.py` |
| Cumulative delay recharged | `envs/delivery_env.py` | `_handle_bus_arrival` | Subtracts previous weighted cost from cumulative totals. | Charge raw incremental waiting/onboard passenger-minutes. | `tests/test_passenger_waiting_increment_accounting.py` |
| Normal dwell misclassified | `envs/dynamics/passenger_dynamics.py` | `process_bus_stop` | Adds boarding/alighting dwell to onboard additional delay. | Passenger service dwell is normal dwell only. | `tests/test_passenger_delay_unit_consistency.py` |
| Extra-dwell exposure overcounts | `envs/delivery_env.py` | `_apply_bus_action` | Uses onboard count after passenger processing and gives all extra dwell to it. | Use onboard count at extra-dwell start; post-extra boarders receive no past delay. | `tests/test_passenger_extra_dwell_exposure.py` |
| Duplicate charging/unloading waiting delay | `envs/delivery_env.py` | `_apply_bus_action` | Adds `waiting * extra_dwell` outside queue clock. | Queue runtime owns waiting passenger-minutes. | `tests/test_passenger_additional_delay_no_double_count.py` |
| Constant station base load | `envs/delivery_env.py` | `_station_load_kw` | Uses config `base_load_kw` for whole episode. | Load from scenario piecewise profile. | `tests/test_station_base_load_profile.py` |
| Power integration misses load boundaries | `envs/delivery_env.py` | `_integrate_station_penalties` | Splits charger boundaries but not base-load profile boundaries. | Split at base, bus, battery, and horizon boundaries. | `tests/test_station_power_interval_integration.py` |
