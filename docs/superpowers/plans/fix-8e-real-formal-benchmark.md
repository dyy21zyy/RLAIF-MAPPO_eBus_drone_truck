# Step 5 Real Formal Benchmark Plan

- Real benchmark execution: replace placeholder rows with `DynamicDeliveryEnv` reset/step rollouts over frozen scenario-bank entries.
- Policy adapters: use deterministic heuristic, assignment PPO, and MAPPO adapters that select feasible masked actions without mutating the environment.
- Formal metric collection: collect metrics from runtime environment instrumentation and fail closed on missing/non-finite/reconciled metrics.
- Diagnostic fixture generation: generate diagnostic scenarios, checkpoints, reward checkpoints, resolved config, and manifests at runtime.
- Temporary artifact strategy: diagnostics write under caller-provided roots such as pytest `tmp_path` or ignored `results/diagnostic/*`.
- Git ignore strategy: ignore generated diagnostic scenario/result/artifact roots and runtime model checkpoint extensions.
- Binary-file PR gate: run staged generated-artifact, binary-extension, Git numstat, MIME, and branch-diff checks before PR creation.
- Explicit staging whitelist: stage only named source, config, test, documentation, and `.gitignore` paths; never use `git add .`.
