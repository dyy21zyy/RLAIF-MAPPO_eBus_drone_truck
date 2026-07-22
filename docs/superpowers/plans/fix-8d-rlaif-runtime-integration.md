# Step 4 Plan — RLAIF Runtime Integration

Starting main SHA: `c4816b44d3f165aa06dcc25734e68485d3350024`.

1. Inspect Phase 5B reward checkpoint creation and legacy runtime scoring.
2. Add RED tests for canonical loading, normalization, event IDs, registry scopes, reward decomposition, policy lineage, and diagnostic smoke rejection.
3. Implement a canonical runtime adapter that delegates checkpoint validation to `load_strict_agent_reward_checkpoint`.
4. Refactor `RewardRegistry` to load assignment-only or four-agent canonical models and fail closed in formal mode.
5. Preserve raw, normalized, clipped, weighted, environment, and combined rewards separately.
6. Add diagnostic and paper configs and a validation CLI.
7. Run focused tests, diagnostic runtime validation, full test suite, compileall, and whitespace checks.
