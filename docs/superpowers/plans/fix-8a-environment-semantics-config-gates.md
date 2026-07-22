# Fix 8a Environment Semantics and Config Gates Plan

Starting main SHA: `17ab5af22711c5bcd1ee8b75a50f496d48e7835d`

## Scope
- Keep passenger bus trip-start events at scheduled timetable times.
- Centralize event priorities and document same-time ordering.
- Use time-varying station base-load profile for candidate projected load and accounting.
- Resolve operational drone/station/battery parameters from runtime entities/config.
- Add formal/smoke config gates for fallback, reward scales, truck-cost consistency, and provenance reporting.

## TDD Plan
1. Add focused RED tests for bus timetable independence, same-time priorities, station load, operational sensitivity, and formal config gates.
2. Run focused tests and record intended failures.
3. Implement minimum coherent fixes.
4. Rerun focused tests, related subsystem tests, and full test suite.
5. Commit and open one draft PR.

## Out of Scope
- No formal training, readiness pilot rollout, ablation/benchmark/sensitivity execution, or final reward-scale estimation.
