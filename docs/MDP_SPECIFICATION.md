# Event-Driven MDP Specification

> **Version:** 1.0 (Stage 3)
> **Implementation:** `envs.delivery_env.DynamicDeliveryEnv`

## Scope and API

The simulator consumes a validated Stage 2 `instance.json` and its CSV/NumPy
artifacts. It has no Gymnasium dependency, but follows the Gymnasium API:
`reset() -> (observation, info)` and
`step(action) -> (observation, reward, terminated, truncated, info)`. A step
resolves one decision and advances all automatic events up to the next decision.
Learning, PPO/MAPPO, preference data, and RLAIF rewards remain out of scope.

## Decision epochs and stable actions

| Agent | Trigger | Action IDs |
| --- | --- | --- |
| Assignment | Parcel release | `0`: TD direct truck; `1..S`: TBD via indexed station; `S+1..2S`: TLD via indexed station |
| Bus | Arrival at an integrated station | Index into `bus.charging_actions_sec` |

`S` is the lexicographically sorted station count. TD means direct truck delivery.
TBD means terminal-to-bus transport followed by station drone delivery. TLD means
truck transport to a station followed by drone delivery. Action masks are part of
every observation. An in-range masked action is deterministically replaced by the
first feasible action and incurs `reward.infeasible_action`; out-of-range and
non-integer actions raise `ValueError`.

## Event model

Events are ordered by `(time_min, priority, insertion_sequence)`. Equal-time
priority is battery completion, drone return, parcel delivery, station parcel
arrival, bus arrival, then parcel release. Parcel releases and first bus arrivals are seeded at reset;
bus continuation, station arrivals, drone returns, and battery completions are
scheduled by transitions. The clock never advances beyond
`bus.delivery_horizon_min`.

TD uses the configured truck queue, loading time, road matrices, customer service,
and optional return to depot. TBD reserves capacity on the earliest freight-enabled
trip that can be loaded before departure. TLD queues a truck to the selected
station. Station parcels occupy locker mass until drone dispatch. A dispatch uses
one drone and one full battery, flies a matrix-distance round trip, and schedules
both drone return and battery recharge completion.

Each bus trip begins at `bus.bus_battery_kwh`. Travel consumes configured energy.
At an integrated station the bus may charge for the selected duration, limited by
charger and station power masks. Charging and freight unloading delay subsequent
arrivals for that trip.

## State, termination, and invariants

Mutable state includes simulation time; parcel status and delayed cost ledger;
truck availability; per-trip state of charge, delay, and freight mass; and each
station's locker mass, drones, batteries, charging sessions, and power limits.
The episode terminates after all events through the delivery horizon are processed.
Every parcel not delivered by then incurs a priority-weighted undelivered cost.
`truncated` is reserved and remains false in Stage 3.

The public `check_invariants()` verifies bounded time, non-negative locker loads,
full batteries, freight loads, and truck availability, plus stable drone-vector
sizes. Stage tests call it after every decision.

## Reward

Rewards are negative weighted costs. Components and coefficients come from the
`reward` configuration: passenger charging delay, bus operating delay, parcel
lateness (minutes times parcel priority), charging energy (kWh), power overload,
bus minimum-SoC violation, locker overflow, monetary truck cost, priority-weighted
undelivered parcels, battery shortage, and infeasible actions. Costs are exposed
individually in `info["cost_components"]`. `rlaif_lambda` is intentionally unused;
learned preference rewards belong to a later stage.
