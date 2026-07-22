# Fix Phase 3 Plan

Starting main SHA: 61b40322b8459ab3bf90d7653faf42962fcca595 (network fetch blocked by CONNECT 403; local branch contains merged Fix Phase 1 PR #13 and Fix Phase 2 PR #14).

Plan:
1. Add RED tests for temporal passenger demand, baseline/effective rate semantics, queue accounting, extra-dwell exposure, station base load, and station-power integration.
2. Implement piecewise time-dependent passenger demand and deterministic baseline truncated-normal sampling.
3. Refactor passenger stop processing to separate normal passenger dwell from delivery-induced extra dwell and expose incremental passenger-minute accounting.
4. Add seeded station base-load artifacts and station-power integration over base/charger boundaries.
5. Wire artifacts into instance manifests, configs, docs, smoke checks, metrics, and environment state where compatible.
6. Run focused tests, existing related tests, smoke tests, full pytest, compileall, diff checks, commit, push, and open draft PR.
