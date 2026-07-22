# Fix Phase 6 Plan

Starting commit: 79e7fc83502504e7e7bcc46e38c07134b834c2bd.

Network fetch of `origin/main` was attempted before editing but the container received `CONNECT tunnel failed, response 403`; work proceeded from the provided repository checkout.

Plan:
1. Add RED tests covering formal policy identity, reward-registry usage, frozen scenario banks, paired hash checks, formal metrics, result schema, ablation/sensitivity lineage, readiness, and resume identity.
2. Implement a formal policy registry that validates explicit algorithms, RLAIF scope, reward agents, checkpoint lineage, and duplicate learned-checkpoint misuse.
3. Implement frozen scenario-bank dataclasses, generation CLI, manifests, hashing, loading, split isolation, and no-regeneration iteration.
4. Refactor paired evaluation to compare scenario IDs plus instance, manifest, and artifact hashes.
5. Add fail-closed metric validation and versioned result rows.
6. Update benchmark, ablation, sensitivity, and readiness commands for formal/smoke separation.
7. Document baseline policies and known infrastructure status without claiming final experiments were run.
8. Run focused tests, smoke bank build, readiness checks, smoke benchmark, full pytest, compileall, diff check, then commit and open a draft PR.
