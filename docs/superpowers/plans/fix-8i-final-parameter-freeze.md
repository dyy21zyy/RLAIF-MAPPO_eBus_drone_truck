# Fix 8i Final Parameter Freeze Plan

Starting main SHA: e2014619e5f0b7a4da1b27800d8fe21945b8c8de

## Preconditions
- `git fetch origin` could not run because this checkout has no `origin` remote configured.
- Local starting branch was `work`; no local `main` branch exists in this checkout.
- Created `codex/fix-8i-final-parameter-freeze` from the available starting commit.

## Implementation plan
1. Inspect existing paper configs, trainers, reward modules, and formal evaluation utilities.
2. Document unresolved, duplicate, and inconsistent parameter sources.
3. Add a canonical final experiment freeze template and method-difference contract.
4. Implement freeze schema validation, placeholder gating, deterministic hashing, and provenance reporting.
5. Implement strict method config difference validation.
6. Add focused tests covering schema, scientific contract, scenario/seed protocol, MAPPO, reward, RLAIF, evaluation, method differences, hashes, and placeholders.
7. Add runtime artifact ignore rules and run focused verification without generating formal artifacts.
