# Baseline Policies

## Truck-direct heuristic

* Assignment: every feasible parcel is assigned to truck-direct; infeasible parcel decisions use deterministic lowest-index fallback and are counted.
* Truck batching: select feasible truck batches by earliest deadline, then release time, then parcel ID.
* Truck route ordering: nearest-neighbor route from depot with deterministic parcel-ID tie breaking.
* Bus freight: buses carry no freight.
* Station drones: stations dispatch no delivery drones unless an explicit future rule requires a rescue dispatch.
* Charging: no discretionary bus charging beyond safety-required depleted-battery handling.

## Integrated rule-based heuristic

* Assignment: prefer feasible bus/locker delivery for parcels whose destination is served by a scheduled stop before deadline; otherwise truck-direct; ties use earliest deadline, then parcel ID.
* Truck batching: batch feasible parcels up to weight/volume limits; order by earliest deadline and route by nearest neighbor.
* Bus loading: greedily load feasible parcels at stops while respecting capacity, deadline, and passenger-service constraints.
* Bus charging duration: charge to the smallest configured duration that preserves minimum SoC through the next route segment.
* Station drone dispatch: dispatch drones only when locker delivery is feasible and a full battery and slot are available.
* Depleted-battery charging: always place depleted batteries into the first available charger using station ID then slot ID tie breaking.
* Passenger-aware charging: charging is skipped when it would create passenger delay beyond configured tolerance.

## Assignment PPO baseline

Assignment PPO is a single assignment-agent policy. Non-assignment decisions are fixed deterministic heuristics: truck route heuristic, greedy/no-freight bus policy as configured, and station battery/drone heuristic. Checkpoint metadata must identify `assignment_ppo` and the heuristic truck, bus, and station policy lineage; it must not be labeled as four-agent MAPPO.

## Formal benchmark deterministic baselines

### Truck-direct heuristic

At assignment decisions the adapter selects a feasible truck-direct (`TD`) candidate whenever the action space exposes one. If no truck-direct candidate is feasible, it falls back to the lowest-index feasible action documented by the environment action mask. Truck, bus, and station decisions use the lowest-index feasible operation; bus decisions do not intentionally load parcel freight for the truck-direct baseline, and station decisions do not dispatch drones unless the environment exposes no safer feasible idle/resource action.

### Integrated rule-based heuristic

The integrated heuristic considers feasible truck-direct, truck-bus-drone, and truck-locker-drone assignment candidates in deterministic priority order. Operational events rank existing environment actions only: truck station/terminal feeders before direct/idle, bus load or charging actions before idle when feasible, and station drone dispatch before idle when feasible. Tie-breaking is always by the action IDs already present in `candidate_actions`; the policy does not create station actions absent from the environment.
