# Fix Phase 5A Preference Dataset Pipeline Plan

Starting main SHA: 78b24d0. Network fetch from GitHub was attempted but blocked by CONNECT 403 in this environment; local HEAD already contains merged Fix Phases 1-4 evidence via merge commits #14-#16 and event/schema constants.

Plan:
1. Add RED coverage for schema validation, original-order target resolution, label-source policy, dataset integrity, duplicate/contradiction handling, grouped splitting, training-only normalization, bus coverage, and feature-order compatibility.
2. Implement `rlaif.preference_schema_v3` as the canonical four-agent preference schema.
3. Refactor reward-pair dataset construction to validate schema, feature order, dimensions, finite values, label sources, duplicate pairs, contradictions, and bus coverage.
4. Refactor grouped splitting to group by state/episode/scenario and emit leakage-safe manifests.
5. Add frozen training-only normalization helpers.
6. Add smoke preference generation and dataset summarization commands.
7. Update formal configs and create smoke configs.
8. Run focused and full verification, commit, push, and open a draft PR.
