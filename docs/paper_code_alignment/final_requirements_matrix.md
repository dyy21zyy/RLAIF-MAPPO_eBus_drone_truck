# Final Requirements Matrix

Status vocabulary: `specified`, `implemented`, `unit-tested`, `integration-tested`, `runtime-validated`, `experiment-validated`, `blocked`.

No item is marked `experiment-validated` because final formal experiments, trained policy checkpoints, validated reward-model checkpoints, real transit ingestion, and final AI preference labels have not been produced.

| Requirement | Status | Evidence / blocker |
|---|---:|---|
| release-time assignment | integration-tested | Parcel release events create assignment decisions at event time. |
| no trip binding at assignment | integration-tested | TBD assignment stores station intent; bus loading binds eligible trip later. |
| truck batching | integration-tested | Truck candidates support multi-parcel batches. |
| truck weight constraints | integration-tested | Invariant tests check onboard truck weight. |
| truck volume constraints | integration-tested | Invariant tests check onboard truck volume. |
| multi-stop routes | integration-tested | Batched truck routes can include multiple stops. |
| physical bus circulation | integration-tested | Trip-to-physical-bus mapping and bus persistence are tested. |
| cross-trip SoC | integration-tested | Physical-bus state carries SoC across trips. |
| cross-trip delay | integration-tested | Physical-bus schedule delay persists across trips. |
| passenger arrivals | integration-tested | Passenger events are generated and consumed by bus-stop processing. |
| boarding/alighting | integration-tested | Passenger stop processing boards and alights without negative counts. |
| passenger-minutes | integration-tested | Passenger waiting/onboard delay metrics are accumulated. |
| bus loading batches | integration-tested | Bus departure exposes loading decisions. |
| charging masks | integration-tested | Charging actions are masked by runtime constraints. |
| drone matching | integration-tested | Station candidates encode drone-to-parcel matching. |
| learned battery charging | integration-tested | Station action selects batteries to charge; no automatic full recharge. |
| charging-slot constraints | integration-tested | Battery charging sessions are capped by slots. |
| station soft power | integration-tested | Overload is a soft cost ledger component, not a hard mask. |
| four-agent asynchronous MAPPO | runtime-validated | Smoke command exercises assignment, truck, bus, and station transitions where PyTorch is available/skipped honestly otherwise. |
| event-time GAE | unit-tested | MAPPO buffer/trainer code discounts by elapsed event time. |
| entity critic | unit-tested | Entity critic networks are covered by MAPPO tests. |
| reward ledger | unit-tested | Reward ledger attribution tests cover cost components. |
| multi-agent preferences | implemented | Schema supports assignment/truck/bus/station preference rows. |
| four reward models | implemented | Four formal reward-model configs exist; validation requires real labels/checkpoints. |
| fail-closed RLAIF | integration-tested | Formal RLAIF config disables fallback and fails on invalid checkpoints. |
| scenario banks | integration-tested | Train/validation/test bank seed ranges are disjoint. |
| paired evaluation | implemented | Benchmark config and policy matrix validation exist; formal outputs blocked. |
| ablations | implemented | Ablation config and runner exist; formal outputs blocked. |
| sensitivity modes | implemented | Sensitivity config and runner exist; formal outputs blocked. |
| artifact provenance | integration-tested | Scenario fixtures include provenance metadata. |
