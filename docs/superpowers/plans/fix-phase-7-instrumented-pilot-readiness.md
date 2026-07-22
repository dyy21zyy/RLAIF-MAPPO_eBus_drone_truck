# Fix Phase 7 Plan — Instrumented Pilot Readiness

Starting main SHA: `5b05de8`.

Status labels used: specified, implemented, unit-tested, integration-tested, runtime-validated, pilot-validated, blocked.

1. Inspect Phase 1–6 artifacts and formal configs — implemented.
2. Add RED tests for readiness validators, MAPPO update evidence, checkpoint round trip, RLAIF gate, and final paper-code contract — unit-tested.
3. Implement diagnostic readiness pilot with manifest, traces, reconciliation reports, MAPPO update report, checkpoint round-trip report, and readiness summary — implemented.
4. Run focused tests, smoke commands, pilot, formal readiness report, and full suite — runtime-validated where commands pass; blocked where formal artifacts are absent.
5. Commit, push, and open a draft PR — specified.
