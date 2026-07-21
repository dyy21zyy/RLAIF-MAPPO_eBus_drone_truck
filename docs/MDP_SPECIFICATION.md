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
explicit per-truck location, availability, remaining capacity, onboard parcel,
distance, travel time, status, and route history; per-trip state of charge, delay, and freight mass; and each
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

## Stage 3 gate compatibility clarification (2026-06-10)

Station power capacity and locker capacity are soft constraints: overload and
overflow remain operationally representable and incur weighted penalties. A bus
charging action is masked only when no physical charger is available; any
resulting power overload is charged by overload energy. `info` exposes cumulative
negative `reward_components`, positive audit `cost_components`, and metrics for
decision counts, delivery counts, drone deliveries, total reward, and corrected
infeasible actions. No learned/RLAIF reward is present in Stage 3.

The hardened event loop integrates station penalties over every elapsed interval.
For each piecewise-constant segment it accumulates overload kW-minutes and locker
overflow kg-minutes, plus the duration for which each violation is positive. Bus
charge endings and drone-battery charge starts/endings split intervals. Locker
mass remains at the station until the explicit drone-dispatch event, so delayed
dispatches contribute their real occupancy duration.

## Stage 7 asynchronous MAPPO interface

The Stage 7 learner consumes the Stage 3 decision sequence without changing its
semantics. Observations expose `agent_id` and `event_type` aliases: `assignment` is
paired only with `PARCEL_ARRIVAL`, `bus` only with integrated-station `BUS_ARRIVAL`,
and terminal observations identify neither active actor. One `step` therefore
creates at most one transition, never a joint action or an inactive-agent row.

Assignment local observations use the existing six assignment features and a
`1 + 2H` mask. Bus local observations use the six bus/station features and a
nine-action mask for charging seconds `[0, 15, 30, 45, 60, 75, 90, 105, 120]`.
Truck and drone behavior remains deterministic. The shared critic state is a
fixed-size aggregate of time, parcel status, decision progress, bus energy/delay/
freight, station locker/battery resources, truck availability, infeasible-action
rate, and terminal flags returned by `get_global_state()`.

The asynchronous transition schema is `agent_id`, `local_obs`, `global_state`,
`action`, `action_mask`, old `log_prob`, old global value, reward, done,
`next_global_state`, `event_type`, `event_time`, and audit `info`. GAE follows this
actual event order and resets at episode boundaries. Initially, each transition
uses the event-to-event environment reward. Optional learned reward is added only
to assignment rows after strict Stage 5 checkpoint loading; bus rows never receive
RLAIF.

## Stage 9 four-agent asynchronous MAPPO code gate

Stage 9 aligns the code surface with the 2026-07-20 Solution Method manuscript.
It exposes assignment, truck, bus, and station decisions in the real event
stream. Each active decision provides candidate actions, candidate features, and action masks, and the environment stores only the active agent row for that
event. There is no inactive-agent padding and no simultaneous joint-action row.

The Stage 9 observation events are `PARCEL_RELEASE` for assignment,
`TRUCK_AVAILABLE` for truck dispatch, `BUS_DEPARTURE` and `BUS_ARRIVAL` for bus
loading/charging, and `STATION_OPERATION` for station drone dispatch and battery
operations. MAPPO uses event-specific candidate-scoring actors with a shared
centralized critic over `get_global_state()`. Rollout storage applies
event-time discounting over elapsed minutes.

This is a code gate, not final experiment evidence. It does not fabricate
preference labels, learned rewards, checkpoints, benchmark results, ablation
results, or sensitivity results.

## Phase 0 final dynamic contract status

Target final architecture: dynamic `PARCEL_RELEASE` assignment chooses only delivery mode plus target station (`TD`, `TBD_<station_id>`, `TLD_<station_id>`); `TRUCK_AVAILABLE` later chooses multi-parcel batches and routes; `ScheduledTrip` is distinct from persistent `PhysicalBusState`; station operation jointly decides drone dispatch and depleted-battery charging starts; multi-agent RLAIF covers assignment, truck, bus, and station agents.

Actual current status: these later-phase behaviors are **specified** by contract and schema only. They are not claimed as implemented, runtime-validated, or experiment-validated in Phase 0.

## Phase 1 dynamic parcel-release and assignment semantics

Phase 1 parcels are initialized as `UNRELEASED`.  Reset schedules one `PARCEL_RELEASE`
event at each parcel `release_time_min`; only that event moves the parcel to
`PENDING_ASSIGNMENT` and exposes the assignment actor.  Unreleased parcels are not
eligible for assignment and are excluded from operational truck, bus-terminal,
station, drone, and pending-parcel queues.

The assignment action space is limited to `TD`, `TBD_<station_id>`, and
`TLD_<station_id>`.  Assignment records only the selected mode, target station,
and assignment time/requirement metadata.  In particular, TBD no longer selects
or reserves a scheduled trip, physical bus, bus departure, truck, route, or drone.
A TBD parcel is first truck-fed to the bus terminal, waits in the generic terminal
queue for its downstream station, and can be loaded by a later freight-carrying bus.
Action-mask construction may estimate feasibility, but must remain side-effect free.

## Phase 2 truck batching

Formal truck decisions at `TRUCK_AVAILABLE` now dispatch bounded multi-parcel batches from the `WAITING_TRUCK` pool instead of a one-parcel task list. Eligible parcels must be released, assigned to TD/TBD/TLD, and not already reserved on an active truck route. Candidate generation is side-effect free and includes idle plus deterministic greedy heuristics for earliest deadline, highest priority, nearest-neighbor geography, bus-terminal consolidation, station consolidation, weight utilization, volume utilization, estimated lateness, and mixed destinations.

Batched routes support `CUSTOMER`, `BUS_TERMINAL`, `INTEGRATED_STATION`, and optional `DEPOT` stops. Applying a batch reserves parcels by moving them to `ONBOARD_TRUCK`; each parcel is unloaded only at its own stop (`DELIVERED`, `AT_BUS_TERMINAL`, or `AT_STATION`). Truck cost is charged once per route from fixed dispatch cost, total distance, and operating time; metrics expose total distance, dispatch count, utilization averages, and parcels per route.

## Phase 3 physical electric-bus circulation

Scheduled trips are timetable service tasks, while `PhysicalBusState` is the persistent vehicle state. Runtime bus operation now keys SoC, current location, schedule delay, next availability, passenger manifest placeholder, and onboard parcels by `physical_bus_id`; trip-indexed structures remain only for stop times, scheduled times, and trip freight manifests.

The physical fleet size is computed as:

`ceil((nominal_one_way_line_time + non_service_relocation_time + minimum_layover_time) / planned_headway)`.

The one-way line time is derived from generated timetable stop times. Relocation and layover assumptions are recorded with `project_extension` provenance in `bus_circulation.json` unless a concrete source is configured. Initial bus energy uses `initial_bus_energy_seed` and samples `Uniform(0.55 * 160, 0.85 * 160)`, so generated SoC values are reproducible and remain in `[88, 136]` kWh for 160 kWh buses.

At each trip start the mapped physical bus is selected from `trip_to_bus.csv`; SoC is not reset. Segment energy uses `segment_distance_km * 1.6`. At terminal completion, SoC and delay persist, relocation energy is subtracted, layover is enforced, and `next_available_time_min` is updated. Complete depletion at or below zero raises severe infeasibility instead of continuing silently. Passenger dynamics remain limited to the existing manifest placeholder pending Phase 4.

## Phase 4 passenger dynamics

Passenger demand is generated before an episode with the configured `passenger_seed`; the simulator never resamples passengers inside a dwell loop.  Each stop receives a seeded truncated-normal baseline rate with mean 0.25 passenger/min, standard deviation 0.10, and bounds [0.05, 0.60].  The effective rate multiplies this baseline by `passenger_demand_intensity` (default 1.0; sensitivity values 0.75, 1.00, 1.25, 1.50, and 2.00 are supported by the generator).  Passenger arrival artifacts include event id, origin, downstream destination, arrival time, and count.

Each stop maintains `waiting_by_destination`, `total_waiting`, `last_queue_update_time`, and `cumulative_waiting_passenger_minutes`.  Before any bus-stop event or dwell-boundary update, waiting passenger-minutes are integrated as current queue size times elapsed minutes.  Physical buses keep destination manifests with capacity 80, so boarding is capacity-capped and passengers alight exactly at their sampled destination; no second binomial alighting process is applied.

Stop service is chronological: integrate arrivals through bus arrival, alight destination passengers, add alighting time (1.5 s/passenger), board up to capacity, add boarding time (3 s/passenger), apply freight unloading and charging, then include pre-generated arrivals during dwell and board them if capacity remains.  The dwell fixed point is bounded to prevent infinite loops.

Passenger delay is waiting passenger-minutes plus onboard additional-delay passenger-minutes, not charging time alone.  Onboard additional delay uses `max(0, realized_dwell - baseline_dwell)` and counts passengers remaining onboard after alighting and after any boarding in that dwell component; normal line-haul travel is not counted as passenger delay.

### Phase 5 bus decisions

The bus actor now uses separate event surfaces for `BUS_TERMINAL_DEPARTURE` and
`BUS_STATION_ARRIVAL`. Terminal departures expose bounded loading-batch
candidates rather than a greedy load-all action. Eligible freight must be
`AT_BUS_TERMINAL`, `TBD`, unreserved, physically present before the departure
cutoff, targeted to a downstream station, and feasible under the hard 20 kg
onboard and 10 kg per-station unloading limits.

Station arrivals expose flash-charging candidates of 0, 15, 30, 45, 60, 75, 90,
105, and 120 seconds. Action 0 is always feasible. Nonzero actions require one
of two physical pantograph chargers and are masked if their 500 kW, 95% efficient
energy addition would exceed the 160 kWh physical bus battery. Station power
capacity remains a soft reward penalty and is surfaced as projected overload.
Passenger-aware dwell includes passenger exchange, parcel unloading at 6
seconds/kg, charging duration, and passenger queues during dwell so equal charge
durations can have different costs for different passenger states.

### Phase 6 station energy operation

Station decisions are now joint `STATION_OPERATION` actions.  A station action may dispatch zero or more available drones to waiting parcels and may start zero or more depleted batteries charging.  Drone resources are explicit (`drone_id`, `home_station_id`, `status`, `available_time_min`, `active_parcel_id`, `active_battery_id`) with statuses `AVAILABLE`, `IN_MISSION`, `RETURNING`, and `TURNAROUND`.  Battery resources are explicit (`battery_id`, `home_station_id`, `status`, `charge_start_time_min`, `charge_completion_time_min`, `assigned_drone_id`) with statuses `FULL`, `IN_USE`, `DEPLETED`, and `CHARGING`.

Battery transitions are controlled by dynamics: dispatch changes `FULL -> IN_USE`, return changes `IN_USE -> DEPLETED`, a selected station action changes `DEPLETED -> CHARGING`, and charge completion changes `CHARGING -> FULL`.  Charging jobs are non-preemptive, last 45 minutes by default, consume 2 kW, and are capped at six simultaneous drone-battery charging jobs per station.  Automatic charging at dispatch or return is removed.

Candidate generation is bounded rather than exhaustive.  The generator always includes idle/no-op and combines heuristic dispatch patterns (earliest-deadline, highest-priority, shortest-mission, minimum-lateness, maximum-cardinality, battery-conservative, and future-capacity-preserving) with a small charging menu (zero, one, fill slots, or reserve-target starts).  Candidates expose dispatch triples, charging battery IDs, estimated delivery/return/lateness, remaining full/depleted batteries, remaining drones, charging-slot use, projected load, projected overload, power margin, expected overload duration, feasibility reasons, heuristic source, and idle flag.

Station power is a soft constraint.  Projected and actual load include station base load, active pantograph bus charging load, and active drone-battery charging load.  Capacity defaults to 1100 kW and overload is exposed and penalized in kW-min rather than hard-masked.

## Phase 7 four-agent asynchronous MAPPO

Phase 7 uses environment reward only. Exactly one active actor creates a transition at each decision epoch: `PARCEL_RELEASE -> assignment`, `TRUCK_AVAILABLE -> truck`, `BUS_TERMINAL_DEPARTURE -> bus`, `BUS_STATION_ARRIVAL -> bus`, and `STATION_OPERATION -> station`. Automatic events create no dummy inactive-agent transitions. Bus loading and bus charging are represented by distinct explicit event embeddings rather than zero padding.

Rewards are recorded in `envs.reward_ledger.RewardLedger` with typed component entries, raw/normalized/weighted amounts, entity/parcel identifiers, source transition IDs, decision-chain references, and provenance. Residual terminal undelivered cost uses documented `terminal_team_distribution` provenance and parcel decision-chain references rather than assigning the whole cost only to the last acting agent.

Event-time GAE uses `gamma_event = gamma ** (elapsed_time / reference_time)` with `gamma=0.997`, `reference_time=1 minute`, and `gae_lambda=0.95`. Advantages are normalized separately within each agent type before PPO updates.
