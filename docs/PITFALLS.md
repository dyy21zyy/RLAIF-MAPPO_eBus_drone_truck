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

## Stage 5 reward-model guardrails

- Fit feature normalization only on the deterministic training split; applying
  independently fitted validation/test statistics leaks information.
- Treat tiny 8/1/1 splits as pipeline smoke tests, not model-quality evidence.
- Never repair an unknown or tied choice, infer a missing preference from action
  features, consume reason text as a first-pass feature, or fall back to rules.
- Preserve `reward_mean` and `reward_std` from all training alternative scores so
  later consumers can use the checkpoint's exact normalization convention.
- Reward-model checkpoints and metric files are runtime artifacts and must remain
  ignored. Stage 5 contains no PPO or MAPPO update path.

### Stage 5 Code Gate versus Runtime Gate

- Do not install PyTorch merely to satisfy the dependency-light Code Gate, and do
  not represent a skipped Runtime Gate as successful training.
- Keep CLI imports dependency-light. Commands that truly require PyTorch must exit
  with the documented installation message when it is absent; PyTorch-only tests
  must skip explicitly.
- Stage 6 smoke tests may disable learned rewards with `rlaif_enabled=false`. Never
  enable RLAIF or silently substitute a heuristic when `reward_model.pt` is missing,
  invalid, or untrained.
- Generated preference JSONL, `results/`, `runs/`, checkpoint directories, and
  `*.pt`, `*.pth`, or `*.ckpt` files are runtime artifacts and must not be committed.

## Stage 6 assignment-PPO guardrails

- Store and optimize only assignment transitions. Bus events must be resolved by
  a fixed, mask-aware baseline and must never enter the PPO buffer.
- Mask assignment logits before constructing/sampling the categorical policy.
  Stage 3 normally keeps `TD` feasible; count and expose any all-zero-mask fallback
  rather than hiding it.
- Do not add a bus actor, parameter sharing, centralized observations, a
  centralized critic, or MAPPO behavior under Stage 6 names.
- With RLAIF disabled, do not touch `reward_model.pt`; total reward is the finite
  environment assignment reward. With RLAIF enabled, fail on a missing or invalid
  checkpoint and apply its saved normalization exactly.
- Candidate objective features and evaluator reason text are not replacement
  rewards. A rule score would be fabricated RLAIF provenance and is prohibited.
- Event-to-event assignment attribution is an initial implementation limitation.
  Do not describe it as optimized delayed ledger credit assignment.
- PyTorch-dependent smoke/tests may skip when PyTorch is absent. `compileall`,
  non-PyTorch tests, documentation checks, and artifact checks must still pass.
- Keep `results/`, `runs/`, checkpoint directories, preference JSONL, and model
  files out of Git. A checkpoint round trip in the smoke test must use a temporary
  directory.
- Final RLAIF-enabled PPO experiments remain blocked until the deferred Stage 5
  Runtime Gate validates a trained reward model.

## Stage 7 asynchronous MAPPO guardrails

- Do not convert the event queue into simultaneous assignment/bus timesteps. Store
  only the actor that actually received the decision event.
- Do not add dummy rewards, no-op transitions, or inactive-agent observations.
- Apply the environment mask before categorical sampling for both actors; bus
  action IDs are indices, not raw charging seconds.
- The centralized critic receives `get_global_state()`; actors receive only their
  local event observation. Do not replace this with QMIX or independent Q-learning.
- Never score bus transitions with the RLAIF model. Never use objective rules or
  evaluator explanations as a learned-reward substitute.
- Disabled RLAIF must not inspect `reward_model.pt`; enabled RLAIF must fail clearly
  if the trained checkpoint is missing or invalid.
- Event-to-event environment reward is acceptable for this first code stage, but
  its cumulative decomposition must not be described as exact per-event causality.
- Keep `results/`, logs, preference data, and `*.pt` checkpoints outside Git. The
  smoke test must use temporary storage and may skip only PyTorch-dependent work.
- Stage 7 Code Gate passing does not unblock final RLAIF-enabled experiments; that
  still requires the Stage 5 Runtime Gate. Do not implement Stage 8 here.

## Stage 8 experiment pitfalls

- A deterministic heuristic is a baseline, not an AI preference label and not reward-model supervision.
- Never replace a missing PPO/MAPPO/reward-model checkpoint with a heuristic while retaining the learned method name; the harness records an explicit skip.
- `rlaif_enabled=false` must not load or require `reward_model.pt`; `true` must validate a real Stage 5 checkpoint.
- Smoke output validates wiring only. It is neither a performance estimate nor a paper-ready table.
- Compare methods only on the identical Stage 2 instance and identical seed list.
- Keep generated `results/`, `runs/`, checkpoints, and preference artifacts out of Git.
