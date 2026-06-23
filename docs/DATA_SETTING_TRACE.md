# Data Setting Trace

This project extends the previous Electric Bus-Drone paper by adding a truck
feeder layer. The experiment scale and most non-truck system settings are
inherited from the previous `eBus-Drone` repository wherever those settings are
available. Real transit inputs take priority for bus stops, trip identifiers,
stop sequences, and planned stop-level timetables. Missing real data are not
described as real; they are inherited from the previous paper setting or marked
as fallback-only for smoke tests.

Source labels:

- `original_ebus_drone`: extracted from `../eBus-Drone`.
- `real_transit_data`: loaded from operator-provided transit CSV files.
- `real_osm_or_geo_data`: optional OSM/geographic data; not the main research data.
- `inherited_default`: documented default used only when the original setting lacks a field.
- `explicit_new_truck_extension`: truck-layer assumption added by this project.
- `fallback_test_only`: committed tiny fixtures or deterministic synthetic data used only for tests/smoke runs.
- `unavailable`: not found in the reference repository or real data source.

| Parameter | New project value | Source | Source file/path in eBus-Drone | Notes |
| --------- | ----------------: | ------ | ------------------------------ | ----- |
| Formal data mode | `original_scale_real_transit` | explicit_new_truck_extension | n/a | New mode preserving original scale while accepting real transit CSVs. |
| Original instance scale | `medium` | original_ebus_drone | `configs/instances/medium.yaml` | Previous-paper medium instance is the default scale anchor. |
| Number of bus stops | `30` target, real route window selected if needed | original_ebus_drone / real_transit_data | `configs/default.yaml` | Real stops are used; if a route is too long, a contiguous window is selected to remain close to 30 stops. |
| Integrated stations | `8` | original_ebus_drone | `configs/instances/medium.yaml` | Selected from real bus stops using inherited count and spacing logic. |
| Number of parcels | `60` | original_ebus_drone | `configs/instances/medium.yaml` | Real parcel demand unavailable; generated from inherited demand scale. |
| Scheduled bus trips | `36` when stop_times must be synthesized | original_ebus_drone | `configs/instances/medium.yaml` | Used only when real stop_times are missing and config explicitly allows synthesis. |
| Freight-carrying trips | `12` when stop_times must be synthesized | original_ebus_drone | `configs/instances/medium.yaml` | Real `freight_allowed` is used when supplied in trips CSV. |
| Planned headway | `10` min if real stop_times are missing | original_ebus_drone | `configs/instances/medium.yaml` | Never used to approximate movement when real stop_times are available. |
| Bus operation horizon | `360` min | original_ebus_drone | `configs/default.yaml` | `generation.bus_operation_horizon_minutes`. |
| Delivery evaluation horizon | `480` min | original_ebus_drone | `configs/default.yaml` | `generation.delivery_evaluation_horizon_minutes`. |
| Bus stops CSV | user-provided `data/raw/transit/real_bus_stops.csv` | real_transit_data | n/a | Contains real `stop_id`, `stop_name`, coordinates, and `stop_sequence`; committed tests use fixture CSVs marked fallback_test_only in documentation. |
| Bus trips CSV | user-provided `data/raw/transit/real_bus_trips.csv` | real_transit_data | n/a | Contains real `trip_id`, `route_id`, `direction_id`, and `service_id` where available. |
| Bus stop_times CSV | user-provided `data/raw/transit/real_bus_stop_times.csv` | real_transit_data | n/a | Formal mode requires this unless synthesis is explicitly enabled. |
| Missing stop_times policy | synthesize only when explicitly allowed | original_ebus_drone | `configs/default.yaml`, `configs/instances/medium.yaml` | Provenance says `real stop_times unavailable; synthesized using original eBus-Drone schedule setting`. |
| Drone speed | `40.0` km/h | original_ebus_drone | `configs/default.yaml` | `drone.speed_kmh`. |
| Drone payload | `4.5` kg | original_ebus_drone | `configs/default.yaml` / `parcel.weight_values_kg` | The reference repo lists parcel weights up to 4.5 kg and drone feasibility uses that scale. |
| Drone service radius | `8.0` km | original_ebus_drone | `configs/default.yaml` | `parcel.drone_service_radius_km`. |
| Max drone round trip duration | `120.0` min | original_ebus_drone | `configs/default.yaml` | `drone.max_round_trip_duration_min`. |
| Drones per station | `3` | original_ebus_drone | `configs/default.yaml` | `drone.drones_per_station`. |
| Initial full batteries | `6` per station | original_ebus_drone | `configs/default.yaml` | `battery.initial_fully_charged_per_station`. |
| Battery charge power | `2.0` kW | original_ebus_drone | `configs/default.yaml` | `battery.charge_power_kw`. |
| Battery charge duration | `45.0` min | original_ebus_drone | `configs/default.yaml` | `battery.charge_duration_min`. |
| Locker capacity | `30.0` kg | original_ebus_drone | `configs/default.yaml` | `parcel.locker_capacity_kg`. |
| Station power capacity | `1100.0` kW | original_ebus_drone | `configs/default.yaml` | `power.station_capacity_kw`. |
| Station base load | average of `80.0` and `180.0` kW | original_ebus_drone | `configs/default.yaml` | Uses inherited nominal range when real station load is unavailable. |
| Bus passenger capacity | `80` | original_ebus_drone | `configs/default.yaml` | `bus.passenger_capacity`. |
| Bus battery capacity | `160.0` kWh | original_ebus_drone | `configs/default.yaml` | `bus.battery_capacity_kwh`. |
| Bus safety battery | `40.0` kWh | original_ebus_drone | `configs/default.yaml` | `bus.safety_battery_kwh`. |
| Bus nominal speed | `30.0` km/h | original_ebus_drone | `configs/default.yaml` | Used for inherited/synthetic schedules only, not real stop_times. |
| Bus freight capacity | `20.0` kg | original_ebus_drone | `configs/default.yaml` | `bus.freight_capacity_kg`. |
| Bus charging actions | `[0,15,30,45,60,75,90,105,120]` sec | original_ebus_drone | `configs/default.yaml` | `charging.action_set_seconds`. |
| Parcel release times | stochastic within inherited bus horizon | original_ebus_drone | `src/data_generation/parcel_generator.py` | Random seed controlled; not real parcel demand. |
| Parcel weights | inherited values/distribution scale | original_ebus_drone | `configs/default.yaml` | Formal builder uses inherited payload-compatible range unless configured otherwise. |
| Deadline mix | tight/moderate/loose `0.30/0.50/0.20` | original_ebus_drone | `src/data_generation/parcel_generator.py` | Slack ranges `20-40`, `40-80`, `80-140` min. |
| Reward weights | mapped from original alpha/eta values | original_ebus_drone | `configs/default.yaml` | Truck/RLAIF extension weights are explicitly new. |
| Truck count | `1` default | explicit_new_truck_extension | n/a | New feeder-layer assumption calibrated to keep scale comparable. |
| Truck capacity | `100.0` kg default | explicit_new_truck_extension | n/a | No equivalent real/reference value found. |
| Truck speed | inherited bus nominal speed unless configured | explicit_new_truck_extension | n/a | Explicit truck extension default, not a real measurement. |
| Truck costs/times | documented defaults | explicit_new_truck_extension | n/a | Configurable; not claimed as previous-paper values. |
| OSM network | optional | real_osm_or_geo_data | n/a | Formal research data do not depend on OSM; haversine/fallback remains available for tests. |
| Smoke transit fixtures | tiny committed CSVs under `tests/fixtures/transit` | fallback_test_only | n/a | Used only by tests and smoke commands; not research data. |
