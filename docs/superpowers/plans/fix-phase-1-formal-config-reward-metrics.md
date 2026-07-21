# Fix Phase 1 implementation plan

Starting SHA: e368873cd1c6c2d93df7cf63b1453ad4e775aaf8

1. Add RED tests covering formal config resolution, reward validation/scaling, status/urgent semantics, formal metrics, and relocation/layover mapping.
2. Implement canonical MAPPO config resolution and validation-only CLIs.
3. Add explicit formal reward blocks and reward scale artifact validation/application.
4. Unify parcel status and urgent field handling.
5. Refactor formal metric collection to use explicit runtime sources.
6. Preserve paper relocation and layover aliases before legacy defaults.
7. Run focused and full verification, commit, and prepare draft PR metadata.
