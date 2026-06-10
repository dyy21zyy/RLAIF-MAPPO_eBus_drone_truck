# Stage 3 Feature Schema

> **Schema version:** 1. Positional order is part of the environment contract.

All feature values are finite floats. Time-like quantities are normalized by the
delivery horizon. Capacity denominators are protected by a minimum value of one.

## Assignment observation

| Index | Name | Unit | Normalization |
| ---: | --- | --- | --- |
| 0 | current time | min | delivery horizon |
| 1 | parcel weight | kg | truck capacity |
| 2 | non-negative deadline slack | min | delivery horizon |
| 3 | parcel priority | integer | maximum synthetic priority (3) |
| 4 | nearest-station drone feasibility | boolean | 0/1 |
| 5 | earliest truck availability | min | delivery horizon |

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

## Stage 4 assignment preference schema (`v1`)

Stage 4 reuses `envs/state_builder.py` for the six-element assignment observation
and all objective action estimates. An assignment JSONL record contains parcel,
system, and station context; all `1 + 2H` named candidates; the Stage 3 action
mask; and a map from action name to objective features. The candidate feature
keys are `action_id`, `action_name`, `feasible_flag`,
`estimated_delivery_time`, `estimated_lateness`, `estimated_truck_distance`,
`estimated_truck_time`, `estimated_bus_wait_time`,
`estimated_bus_linehaul_time`, `estimated_drone_time`,
`estimated_locker_load_after_assignment`, `estimated_station_power_margin`, and
`infeasibility_reasons`. No preference score is part of this schema.

Prompt records use `prompt_version=v1` and store the state/pair IDs, prompt text,
and pair-selection metadata. Validated preference records add evaluator model,
temperature, parser/validation status, raw response, and training usability;
confidence below `0.6` is retained but unusable by default.
