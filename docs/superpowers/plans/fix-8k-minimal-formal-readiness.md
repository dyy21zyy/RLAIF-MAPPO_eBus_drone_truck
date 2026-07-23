# Minimal formal readiness implementation plan

Starting main SHA: `9a7a6b22b96e084b1de2e3b1c4b3ba6cc913a1e7` (remote fetch was blocked by the execution network; this is the repository HEAD available in the workspace).

1. Reuse the production scenario-bank builder to add a thin formal input preparation command that generates train/validation/test banks, validates split disjointness and hashes, estimates reward reference scales from the train bank only, and writes ignored runtime manifests/configs under `results/formal/`.
2. Complete the assignment-only and all-agent formal RLAIF-MAPPO YAMLs by copying the executable environment MAPPO baseline fields and changing only intended RLAIF scope/model/output fields.
3. Add a formal Assignment PPO config and minimal CLI/lineage support needed for validate-only, seed overrides, output roots, checkpoint lineage, and reload validation.
4. Add focused tests for generated-input validation, runtime config placeholder rejection, config equality/scopes, Assignment PPO lineage/reload, and preformal subprocess failure propagation.
5. Execute preparation, reduced training checks where possible, compile/test/diff checks, then commit source-only changes.
