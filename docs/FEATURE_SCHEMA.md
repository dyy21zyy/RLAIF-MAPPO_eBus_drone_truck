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
