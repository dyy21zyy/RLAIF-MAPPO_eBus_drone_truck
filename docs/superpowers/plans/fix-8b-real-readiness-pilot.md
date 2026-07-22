# Fix 8b Real Readiness Pilot Plan

Starting main SHA recorded from the local repository: `f21afedf9452e82d31a62db00ed49a22248ac610`.

1. Replace synthetic trace and checkpoint generation with a `DynamicDeliveryEnv` diagnostic rollout.
2. Use `training.mappo_trainer.collect_episode`, `AsyncMAPPOBuffer`, event-time GAE, `update_mappo`, `save_checkpoint`, and `load_checkpoint`.
3. Add runtime export interfaces for event, bus, passenger, station-power, truck, parcel, and reward-ledger traces.
4. Add focused tests that assert real rollout artifacts, event coverage, optimizer updates, event-time GAE, reconciliation reports, checkpoint round trip, and RLAIF blocking status.
5. Run the diagnostic pilot, focused tests, related MAPPO/environment tests, full pytest, compileall, diff check, commit, push, and open a draft PR.
