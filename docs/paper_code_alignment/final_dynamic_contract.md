# Final Dynamic Multi-Agent Contract (Phase 0)

Status: **specified**. This document freezes the target paper-code contract for later phases; it does not claim that truck batching, physical-bus circulation, passenger dynamics, station battery decisions, or multi-agent RLAIF are implemented, runtime-validated, or experiment-validated.

## Dynamic parcel assignment
Parcels arrive dynamically. At each `PARCEL_RELEASE` event, the assignment agent chooses only `TD`, `TBD_<station_id>`, or `TLD_<station_id>`. The assignment agent selects delivery mode plus target station. It must not select a specific truck, a specific scheduled bus trip, a specific physical bus, a truck batch, a truck route, a bus loading batch, a drone, or a battery-charging schedule.

## Truck decisions
At `TRUCK_AVAILABLE`, the truck agent is specified to choose a multi-parcel batch and multi-stop route. A candidate batch may mix direct-delivery customers, bus-terminal feeder parcels, integrated stations, and other mixed destinations. Later phases must enforce both weight and volume constraints.

## Physical buses and scheduled trips
A scheduled trip is not a physical bus. Later implementation must include `trip_to_bus: dict[str, str]`. Physical bus energy, location, delay, passengers, parcels, and availability persist across scheduled trips.

## Passenger delay
Passenger delay is specified as waiting passenger-minutes plus onboard additional-delay passenger-minutes. It must not be defined as charging duration alone.

## Station operation
At `STATION_OPERATION`, the station agent is specified to jointly decide drone-parcel dispatch matching and which depleted batteries begin charging. Drone-returned batteries become depleted and do not start charging automatically.

## Multi-agent RLAIF
The final RLAIF scope is `RLAIF_AGENT_TYPES = {"assignment", "truck", "bus", "station"}`. The target system will use agent-aware preference data and agent-aware reward models.
