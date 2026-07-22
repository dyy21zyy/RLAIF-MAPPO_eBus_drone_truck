# Requirements Traceability v2

| Requirement | Phase 0 status | Notes |
| --- | --- | --- |
| dynamic release-time assignment | specified | Assignment action space is frozen to `TD`, `TBD_<station_id>`, and `TLD_<station_id>`. |
| truck batching | specified | Later phases implement multi-parcel batch and route selection. |
| physical-bus circulation | specified | `ScheduledTrip` and `PhysicalBusState` are distinct; `trip_to_bus` is required later. |
| passenger dynamics | specified | Delay uses waiting and onboard passenger-minutes, not charging duration alone. |
| station battery decisions | specified | Charging starts are learned station decisions, not automatic. |
| multi-agent RLAIF | specified | Agent-aware preference data and reward models are deferred. |
| formal parameter schema | specified | `configs/paper/base_*.yaml` and validators define Phase 0 schema. |

| Fix Phase 2 stop-by-stop physical bus event chain | unit-tested | Physical buses operate scheduled passenger trips stop by stop; ordinary passenger service, integrated-station charging, causal downstream arrivals, segment energy, relocation and layover are covered by focused tests. |
