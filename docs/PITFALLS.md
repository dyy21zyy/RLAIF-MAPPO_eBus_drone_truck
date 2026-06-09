# Pitfalls and Guardrails

## Scope creep

- Do not place simulator behavior in Stage 1 placeholders.
- Do not treat training configuration templates as implemented trainers.
- Do not add PPO, MAPPO, preference modeling, or RLAIF before their stage gate.

## Reproducibility

- Record every seed and configuration path.
- Seed before generating data, initializing environments, or creating models.
- A Python hash seed set at runtime affects child processes, not the already
  running interpreter; launch-level hash determinism may require setting
  `PYTHONHASHSEED` before Python starts.
- Deterministic seeds do not by themselves guarantee deterministic GPU kernels.

## Configuration

- Keep units in field names (`_km`, `_min`, `_kg`, `_kw`, `_kwh`, `_sec`).
- Treat configuration as experiment input; avoid hidden constants.
- Validate malformed and missing configuration files with actionable errors.

## Logging and artifacts

- Avoid duplicate handlers in notebooks or repeated experiment setup.
- Keep generated logs, outputs, and checkpoints out of source modules.
- Never depend on internet access in smoke tests.

## Data and later stages

- Use only approved public, synthetic, or user-provided route/order data.
- Keep a small deterministic fallback instance for CI and offline use.
- Validate physical units and state invariants before training an agent.

## Stage-gate remediation: empty configurations

- **Issue:** The dependency-free YAML path accepted a blank file as an empty
  dictionary, while the public loader contract required empty configurations to
  raise `ConfigError`.
- **Root cause:** `_load_simple_yaml` naturally returns an empty mapping when it
  sees no mapping entries, and `load_config` rejected only `None` and non-mapping
  roots—not empty mappings.
- **Fix:** Reject whitespace-only input before parser selection and reject an
  empty root mapping after either parser returns. Missing files, malformed input,
  and non-mapping roots continue to raise `ConfigError`.
- **Prevention:** Keep parser-independent validation around parser selection, and
  test missing, blank, empty-mapping, malformed, non-mapping, and valid inputs.
  The fallback parser remains intentionally limited to two-space nested mappings
  and inline scalar lists; use PyYAML for advanced YAML syntax.
