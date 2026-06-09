# MDP Specification Template

> Status: reserved for Stage 3. No simulator is implemented in Stage 1.

## Scope

Document the event-driven decision process for truck, electric-bus, integrated
station, and drone delivery. The implemented environment must cite the exact
version of this specification that it follows.

## Decision epochs

| Decision agent | Triggering event | Action family | Planned stage |
| --- | --- | --- | --- |
| Assignment | Parcel arrival | TD, TBD station, or TLD station | Stage 3 |
| Bus | Integrated-station arrival | Discrete charging duration | Stage 3 |

## State

Define parcel, vehicle, station, queue, battery, power, time, and capacity state.
Include units, bounds, initialization, and terminal-state behavior.

## Action and feasibility

Define stable action indices, action masks, infeasible-action correction, and the
mapping between action IDs and physical decisions.

## Transition model

Specify event priorities, tie-breaking, automatic events, resource updates, and
time-integrated power/locker quantities.

## Reward

List each environment cost, sign convention, units, coefficient, timing, and
parcel-level delayed reward ledger. RLAIF terms must remain separate until their
later implementation stage.

## Termination and invariants

Document the delivery horizon and invariants such as non-negative capacities,
battery counts, and locker loads.
