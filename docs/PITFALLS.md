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

## Stage 2 data-pipeline guardrails

- Keep the fallback path independent of internet access and optional OSM tooling.
- Never replace synthetic parcels with scraped private order data, and do not
  scrape sites whose terms or robots policy prohibit it.
- Treat the fallback grid and generated corridor as smoke-test fixtures, not as
  validated representations of Shanghai traffic or public-transport operations.
- Preserve explicit physical units in CSV columns and instance metadata.
- Matrix consumers must use `instance.json` or `instance.yaml` index metadata;
  row order must not be inferred from CSV sorting.
- Full mode can legitimately degrade to fallback mode when `osmnx`, internet
  access, or an OSM response is unavailable; inspect manifest warnings and mode.
- Stage 2 produces static instance data only. Do not add event progression,
  dispatch decisions, rewards, PPO/MAPPO, preference data, or reward models here.

## Stage 2 gate remediation: JSON-compatible YAML

- **Issue:** The first Stage 2 smoke run built every artifact but failed while
  loading `instance.yaml` in an environment without PyYAML.
- **Root cause:** JSON is valid YAML and was used to keep manifest serialization
  dependency-light, but the Stage 1 fallback YAML parser supported only nested
  `key: value` mappings and inline scalar lists.
- **Minimal fix:** Detect JSON-looking documents and parse them with the standard
  library before selecting PyYAML or the simple YAML parser.
- **Regression prevention:** The Stage 2 offline instance-loading test reads both
  manifests, and the full Stage 1 suite continues to test malformed, blank,
  empty-root, non-mapping, and valid configuration behavior.

## Stage 3 gate remediation before Stage 4

- The required gate entry point was missing because the original smoke module was
  named `smoke_test_environment`; keep `experiments.smoke_test_env` as the stable
  gate command and require `--fallback`.
- Station power capacity is a soft constraint, not an action-mask hard constraint.
  Charging remains selectable when a charger is free, and overload energy is
  penalized through the environment reward.
- Keep cumulative negative `reward_components` and named sanity `metrics` in
  `info`; retain positive `cost_components` only as an audit-compatible alias.

## Stage 4 preference-data guardrails

- Objective estimates may select which pairs to ask about, but must never become
  labels or preference scores.
- Offline mode must remove stale preference output rather than leave a file that
  could be mistaken for newly generated labels.
- Never silently discard malformed JSON, unknown actions, reversed rejected
  actions, out-of-range confidence, or unmatched replay labels; preserve them in
  the failed JSONL output.
- Low confidence is not invalid data. Retain it with
  `usable_for_training=false` below the `0.6` default threshold.
- Generated preference data is reproducible runtime output and is ignored by
  Git; the code, schemas, prompts, tests, and provenance documentation are the
  committed Stage 4 artifacts.

### API configuration and label provenance

- Empty API configuration is acceptable in a development checkout. Keep private
  keys out of source control and provide them through the environment variable
  selected by `rlaif.api_key_env` (normally `RLAIF_API_KEY`). Base URL and model
  may come from `RLAIF_API_BASE_URL` / `RLAIF_MODEL_NAME` or the config file.
- Missing API settings must stop API mode; they must never activate a fallback
  based on earliest deadline, shortest distance, lowest cost, feasibility, or
  another candidate feature.
- Objective candidate features are valid state/action context and may be shown to
  an evaluator. They must never be converted directly into `chosen`/`rejected`.
- Offline mode creates blank manual-label templates and removes stale preference
  output. Null template fields are not labels and must pass through replay
  validation only after a user or external evaluator fills them.
- Replay validates only records in the explicitly supplied file. Never infer a
  label for a missing prompt or repair an invalid choice with a heuristic.
- Reward-model training must stop when no valid record has
  `usable_for_training=true`; do not substitute rules or synthetic preferences.
