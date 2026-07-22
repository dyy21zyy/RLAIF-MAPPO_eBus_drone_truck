# Fix 8c scenario-bank multi-scenario training plan

Starting main SHA: `8cc8cefb93cb8145f6e0283a8b80f05c5d00b9e1` (local repository has no `origin` remote, so latest remote main could not be fetched in this container).

1. Add canonical `ScenarioSeedTuple` and propagate overrides into the existing `build_instance()` path.
2. Freeze every artifact referenced by `instance.json`, rewrite paths to scenario-relative names, and compute non-volatile content hashes.
3. Extend scenario-bank manifests with split, seed, dynamic/static artifact uniqueness, and bank hashes.
4. Add deterministic scenario sampling and a factory that validates frozen hashes before creating fresh `DynamicDeliveryEnv` instances.
5. Refactor MAPPO training to use scenario banks when configured, log per-episode scenario provenance, and store checkpoint lineage.
6. Add focused tests for seeds, reproducibility, frozen scenarios, bank isolation, samplers, factory, MAPPO provenance, checkpoint lineage, and test-bank rejection.
7. Build tiny diagnostic banks and run short multi-scenario MAPPO verification; do not run formal large banks or long training.
