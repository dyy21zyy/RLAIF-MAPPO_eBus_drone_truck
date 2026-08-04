# Implementation Plan: Assignment Preference Quality Gate v2

## Commit 1 — quality contracts and pure gate
- Create `rlaif/preference_quality.py`: versions, aliases, modes/resources, metric registry, feature extraction, evidence/direction/applicability, lexical semantics, dominance, canonical reason, legacy audit, aggregation, coverage and proportional allocation.
- Create `tests/test_preference_quality.py` with focused synthetic mode, criteria, direction, semantics, dominance, trade-off, determinism, and sampling contracts.
- Verify: `python -m pytest -q tests/test_preference_quality.py`.

## Commit 2 — offline migration
- Create `experiments/audit_assignment_preferences.py` with strict input/hash/count/identity validation and accepted/quarantine JSONL plus JSON/Markdown output.
- Create `tests/test_assignment_preference_quality_pipeline.py` proving input immutability, safe migration, quarantine, label preservation, and zero calls.
- Verify CLI help and synthetic fixture.

## Commit 3 — generation and cache integration
- Modify `experiments/generate_formal_multiagent_preferences.py` to resolve versions by agent, emit grounded assignment prompt v2, validate assignment v2 before counting, preserve raw reason, write deterministic reason, and separate cache identities/provenance.
- Modify `configs/paper/rlaif_preference_generation.yaml` with exact assignment versions/gates, quality flags, pair floors, Scheme A environment contract, target 1600 and budget 3000.
- Update generation tests where assignment v2 is explicitly configured while preserving non-assignment v1 behavior.

## Commit 4 — Phase 2 gate and provenance
- Modify `experiments/run_hard_locker_reexperiment.py` Phase 2 command/order, outputs, validation, marker lineage, and resume invalidation to require legacy and final quality audits before training.
- Update `tests/test_hard_locker_reexperiment.py`, `tests/test_hard_locker_resume.py`, and `tests/test_phase2_preference_count_consistency.py` with dry-plan and stale-marker contracts.

## Commit 5 — documentation and verification
- Save the supplied specification in `docs/specs/assignment_preference_quality_spec.md` and document Scheme A operations/environment variables in experiment documentation.
- Run compileall, all focused suites, CLI help, synthetic audit, Phase 2 dry plan, `git diff --check`, and broader tests. Record baseline-only failures. Confirm no production API calls and untouched existing roots.
- Commit clean work, push `codex/assignment-preference-quality-gate-v2`, and open the requested PR with exact outcomes and later-real-run requirements.
