# Stage 3 Feature Schema

> **Schema version:** 1. Positional order is part of the environment contract.

All feature values are finite floats. Time-like quantities are normalized by the
delivery horizon. Capacity denominators are protected by a minimum value of one.

## Assignment observation

The ordered global/parcel prefix has 17 values:
`time_norm`, `deadline_remaining_norm`, `weight_norm`, `volume_norm`, the three
priority one-hot values, `drone_feasible_global`, depot/customer travel time,
nearest-station distance, idle-truck count, earliest truck availability, average
truck capacity remaining, terminal queue length, next freight-bus arrival,
feasible freight-bus count, and average remaining bus freight capacity.

Each lexicographically sorted station then contributes 10 values: customer
distance, drone round-trip time, station-specific drone feasibility, locker
remaining capacity, locker occupancy, idle drones, full batteries, power margin,
next feasible bus wait, and remaining bus freight capacity to that station. The
total length is therefore `17 + 10S`. Exact machine-readable names are exported by
`envs.state_builder.assignment_feature_names()`.

The current Stage 2 data has no vehicle volume constraint, so parcel volume is
normalized by the maximum parcel volume in the instance. The one-parcel-per-trip
truck policy restores capacity after each dispatch, so average remaining truck
capacity is represented as full when a truck exists. These are explicit stable
approximations, not omitted fields.

The assignment mask has `1 + 2S` entries in stable TD, TBD-station, TLD-station
order. Station order is lexicographic by `station_id`.

## Bus observation

| Index | Name | Unit | Normalization |
| ---: | --- | --- | --- |
| 0 | current time | min | delivery horizon |
| 1 | trip state of charge | kWh | bus battery capacity |
| 2 | accumulated trip delay | min | delivery horizon |
| 3 | station locker load | kg | locker capacity |
| 4 | station full batteries | count | initial full batteries |
| 5 | bus freight load | kg | bus freight capacity |

The bus mask length equals `len(bus.charging_actions_sec)` and action zero is
always feasible. Terminal observations use agent `terminal`, feature `[1.0]`, and
an empty mask.

## Observation envelope

Every decision observation is a dictionary with `agent`, `entity_id`, `time_min`,
`features`, and `action_mask`. Assignment entity IDs are parcel IDs. Bus entity
IDs are `trip_id:station_id`. Global MAPPO state and padding remain deferred until
the MAPPO stage.

## Stage 4 assignment preference schema (`v2`)

Stage 4 reuses `envs/state_builder.py` for the `17 + 10S` assignment observation
and all objective action estimates. An assignment JSONL record contains parcel,
system, and station context; all `1 + 2H` named candidates; the Stage 3 action
mask; and a map from action name to objective features. Human-readable fields are
`action_id`, `action_name`, and `infeasibility_reasons`. The ordered numeric
features are the TD/TBD/TLD one-hot values, normalized 1-based station index,
feasibility flag, and normalized delivery time, lateness, truck distance, truck
time, bus wait, bus linehaul, drone time, locker load after assignment, and
station power margin. No preference score is part of this schema.

Prompt records use `prompt_version=v1` and store the state/pair IDs, prompt text,
and pair-selection metadata. Validated preference records add evaluator model,
temperature, parser/validation status, raw response, and training usability;
confidence below `0.6` is retained but unusable by default.

## Stage 5 reward-model input schema (`v2`)

Stage 5 joins each valid preference to its Stage 4 `state_id`. Both alternatives
reuse the expanded `assignment_features` vector. Per-action input is the
following 14-value ordered numeric vector from
`candidate_action_features[ACTION_NAME]`: `action_type_TD`, `action_type_TBD`,
`action_type_TLD`, `action_station_index_norm`, `feasible_flag`, followed by the
nine normalized objective estimates listed above. `action_id` is embedded
separately. Textual
reasons and `infeasibility_reasons` are not model inputs.

State and action means and population standard deviations are fitted using both
alternatives in the training split only. Validation and test data reuse those
statistics with an epsilon of `1e-6`; non-finite normalized values are rejected.
