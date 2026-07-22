# Fix Phase 2 plan

Starting main SHA: 9068ad3.

1. Confirm reset-time integrated-station scheduling defects with focused tests.
2. Add canonical automatic bus event constants and structured runtime trace.
3. Replace reset-time bus arrivals with trip-start events for every scheduled trip before 360 minutes.
4. Generate downstream stop arrivals only after actual departures, using physical buses as runtime state.
5. Preserve physical-bus SoC, freight, passengers, location, delay, relocation and layover across trips.
6. Update metrics, invariants, smoke diagnostics and documentation.
7. Run focused tests, related tests, smoke commands, full pytest, compileall, diff checks.
