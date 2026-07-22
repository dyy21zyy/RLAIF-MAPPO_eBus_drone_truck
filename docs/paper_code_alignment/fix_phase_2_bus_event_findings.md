# Fix Phase 2 bus event findings

Starting main SHA: 9068ad3.

| Defect | file | function | current behavior | required behavior | failing test |
|---|---|---|---|---|---|
| Integrated-station-only progression | `envs/delivery_env.py` | `reset` | Scheduled the first integrated-station `bus_arrival` at reset. | Schedule only one trip-start event for every pre-360 scheduled trip. | `tests/test_bus_all_stop_progression.py` |
| Ordinary stops skipped | `envs/delivery_env.py` | `_handle_bus_arrival` | Required `station_id = stop_to_station[stop_id]`, so ordinary stops could not be processed. | Process ordinary stops without station decisions. | `tests/test_bus_all_stop_progression.py` |
| Non-freight trips not fully operated | `envs/delivery_env.py` | `reset` | Loading decisions were created only for freight trips and arrivals only targeted integrated stations. | Every passenger trip before the operation horizon starts and visits all stops. | `tests/test_non_freight_passenger_trips.py` |
| Downstream arrivals pre-scheduled independently | `envs/delivery_env.py` | `reset` | First integrated arrivals were in the queue before terminal dwell/loading. | Create arrivals only after actual departure. | `tests/test_bus_departure_arrival_causality.py` |
| Loading delay not propagated | `envs/delivery_env.py` | `_apply_bus_loading_action` | Loading updated delay but did not shift an already queued first arrival. | First arrival is departure after loading plus segment running time. | `tests/test_bus_loading_delay_propagation.py` |
| Previous-trip delay not fully propagated | `envs/delivery_env.py` | `bus_departure` handler | Next trip could be rescheduled while stale trip arrivals remained. | Reschedule only trip start until bus availability; no downstream events. | `tests/test_bus_no_stale_arrival_events.py` |
| Trip-keyed mirrors acting as source of truth | `envs/delivery_env.py`, `envs/state_builder.py` | bus handlers/observations | Trip dictionaries mirrored and sometimes drove SoC/delay/freight. | Physical bus and runtime trip state own behavior; trip dicts are reporting views. | `tests/test_bus_trip_completion_and_relocation.py` |
| Segment energy only between integrated stations | `envs/delivery_env.py` | `_handle_bus_arrival` | Energy was deducted only for arrivals at integrated stations. | Every consecutive stop segment consumes energy once. | `tests/test_bus_segment_energy_accounting.py` |
