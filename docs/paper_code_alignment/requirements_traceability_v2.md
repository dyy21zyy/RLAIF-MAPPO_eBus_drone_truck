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

## Fix Phase 6 formal evaluation integrity

Formal evaluation now uses frozen scenario-bank manifests. All methods share identical scenario artifacts, and paired comparisons validate scenario ID, instance hash, scenario-manifest hash, and exogenous artifact hashes before comparison. Environment MAPPO, assignment-only RLAIF-MAPPO, and full RLAIF-MAPPO are separate formal method identities with separate policy checkpoints. Full RLAIF evaluation requires four agent-specific reward checkpoints loaded through `RewardRegistry`; assignment-only RLAIF enables only the assignment reward model. Reward models do not select evaluation actions; they validate lineage and score selected transitions for decomposition only.

Formal metrics are fail-closed: missing instrumentation is missing, not zero. Legitimate zero values require an instrumented source and explicit legitimate-zero provenance. Ablations that require retraining require separate checkpoints and actual configuration differences. Sensitivity experiments distinguish fixed-policy robustness from retrained-policy sensitivity and do not aggregate the two modes together by default.

This infrastructure does not claim that the final 100-scenario, three-seed paper benchmark has been executed; formal readiness remains blocked until final frozen banks, trained policies, and validated formal reward checkpoints exist.
