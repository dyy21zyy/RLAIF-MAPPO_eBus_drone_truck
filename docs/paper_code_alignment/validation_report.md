# Paper-Code Alignment Validation Report

Date: 2026-07-11 through 2026-07-21

## Environment

| Item | Result |
| --- | --- |
| Python | `Python 3.12.4` in the earlier runtime; current local gate uses the active Python environment |
| PyTorch | PyTorch runtime unavailable in the 2026-07-11 gate; available in the 2026-07-20 Stage 9 local gate |
| PyTorch failure mode | Earlier installed package raised Windows DLL initialization failure while importing `torch` |
| PyTorch runtime gates | Optional-torch tests keep unavailable runtimes as explicit skips or runtime-gate failures instead of import crashes |
| Runtime boundary | Final RLAIF-enabled experiments remain deferred until real labels, trained checkpoints, and configured benchmark runs exist |

## Current Status

Implementation is complete through the Stage 9 four-agent asynchronous MAPPO
code gate for the 2026-07-20 Solution Method alignment. No preference labels, learned rewards, checkpoints, or final results are fabricated.

## Baseline Findings

| Command | Result | Classification | Notes |
| --- | --- | --- | --- |
| `python -m pytest -q` | failed before 2026-07-11 fixes | code gate | Test collection crashed on broken PyTorch imports. |
| `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml` | failed before 2026-07-11 fixes | code gate | `seed_everything()` attempted to import broken PyTorch. |
| `git diff --check` | passed before fixes | formatting gate | No whitespace errors at baseline. |

## Implemented Validation

| Command | Result | Classification | Notes |
| --- | --- | --- | --- |
| `python -m pytest tests/test_optional_torch.py -q` | `4 passed` | focused code gate | Verifies broken PyTorch is treated as unavailable. |
| `python -m pytest tests/test_optional_torch.py tests/test_assignment_ppo.py tests/test_mappo_async.py tests/test_reward_model_wrapper.py tests/test_stage5_code_gate.py -q` | `18 passed, 4 skipped` | focused code gate | Verifies Stage 5/6/7 optional PyTorch behavior. |
| `python -m pytest tests/test_paper_code_alignment_docs.py -q` | `1 passed` | focused docs gate | Verifies traceability, validation, decision, and plan records. |
| `python -m experiments.smoke_test_assignment_ppo` | `SKIP: PyTorch is unavailable` | code gate skip | Earlier runtime skip without traceback. |
| `python -m experiments.smoke_test_mappo_async` | `SKIP: PyTorch is unavailable` | code gate skip | Earlier runtime skip without traceback. |
| `python -m experiments.smoke_test_reward_model` | returns runtime-required message | runtime gate skip | Expected nonzero Runtime Gate result without PyTorch. |
| `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml` | passed | dependency-light smoke | Reports seeded backend availability. |
| `python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback` | passed | dependency-light smoke | Validates fallback artifact set. |
| `python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback` | passed | dependency-light smoke | Completes one event-driven episode with invariants passed. |
| `python -m experiments.smoke_test_experiments` | passed | dependency-light smoke | Learned checkpoints are skipped when missing. |
| `python -m experiments.smoke_test_original_scale_real_transit_data --config configs/original_scale_real_transit.yaml` | passed | dependency-light smoke | Validates fixture-backed source-aware original-scale data mode. |
| `python -m experiments.smoke_test_original_scale_real_transit_env --config configs/original_scale_real_transit.yaml` | passed | dependency-light smoke | Runs the original-scale event-driven environment smoke. |
| `python -m experiments.smoke_test_original_scale_real_transit_rlaif --config configs/original_scale_real_transit.yaml` | passed | dependency-light smoke | Confirms prompts are created and labels are not fabricated. |
| `python -m pytest -q` | `132 passed, 6 skipped` | full code gate | Full 2026-07-11 test suite passed after optional PyTorch fixes. |
| `python -m compileall -q .` from `work/rlaif-align` junction | passed | compile gate | Short path avoids Windows `.pyc` path-length failure. |
| `git diff --check` | passed | formatting gate | Only line-ending warnings were emitted by Git. |
| `python C:\Users\dyy21\.codex\skills\rlaif-mappo-paper-alignment\scripts\check_traceability.py .` | passed | alignment docs gate | Confirms required IDs and no-fabrication wording. |
| `python C:\Users\dyy21\.codex\skills\.system\skill-creator\scripts\quick_validate.py C:\Users\dyy21\.codex\skills\rlaif-mappo-paper-alignment` | passed | skill gate | The reusable skill is structurally valid. |

## Stage 9 Verification Log

| Command | Status | Evidence |
| --- | --- | --- |
| `python -m pytest tests/test_paper_alignment_traceability.py -q` | pass | Initial Stage 9 traceability docs were created and validated. |
| `python -m pytest tests/test_four_agent_environment.py tests/test_stage3_environment.py -q` | pass | Four active agent ids and Stage 3 regressions passed during Task 3. |
| `python -m pytest tests/test_mappo_buffer.py -q` | pass | Four-agent buffer and event-time discounting passed during Task 4. |
| `python -m pytest tests/test_mappo_networks.py -q` | pass | Candidate-scoring actor and actor registry passed during Task 5. |
| `python -m pytest tests/test_mappo_async.py tests/test_mappo_networks.py tests/test_mappo_buffer.py -q` | pass | Stage 9 trainer smoke contract and MAPPO unit tests passed during Task 6. |
| `python -m pytest tests/test_paper_alignment_traceability.py tests/test_four_agent_candidate_schema.py tests/test_four_agent_environment.py tests/test_mappo_buffer.py tests/test_mappo_networks.py tests/test_mappo_async.py -q` | pass | Final focused Stage 9 gate: 24 passed. |
| `python -m pytest -q` | pass | Final Stage 9 full suite before strict-latest merge: 146 passed, 3 skipped, 1 known reward-model smoke warning. |
| `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml` | pass | Stage 1 smoke passed. |
| `python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback` | pass | Stage 2 fallback data-pipeline smoke passed. |
| `python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback` | pass | Stage 3 event-driven environment smoke passed. |
| `python -m experiments.smoke_test_experiments` | pass | Stage 8 smoke passed with five baseline successes and missing learned checkpoint skipped. |
| `python -m experiments.smoke_test_mappo_async` | pass | Stage 9 smoke collected assignment=60, truck=7, bus=24, station=1 transitions. |
| `python -m compileall -q .` | pass | No syntax errors. |
| `git diff --check` | pass | No whitespace errors. |

## Strict-Latest Merge Verification

This merge incorporates old-repository `main` at
`ddd81a0a96fcfc03e14a04a6c06c48fd77512769` into the four-agent Stage 9 head
`c69144575d0495dd1203e086e7dc0afd574f582d`.

Final full-suite verification was rerun from the short checkout
`C:\Users\dyy21\Documents\Codex\rlsm_latest` because the deep Codex workspace
path can exceed Windows path-length limits when tests generate nested temporary
artifacts. The earlier deep-path failures were environmental `WinError 206` or
temporary-directory permission errors, not code-gate failures.

| Command | Status | Evidence |
| --- | --- | --- |
| `python -m pytest tests/test_paper_alignment_traceability.py tests/test_paper_code_alignment_docs.py tests/test_mappo_networks.py -q --basetemp .pytest-tmp` | pass | Strict-latest focused docs/network gate: 9 passed. |
| `python -m pytest -q --basetemp work/pytest-final-tmp` | pass | Strict-latest full suite from the short checkout: 152 passed, 3 skipped, 1 reward-model smoke warning. |
| `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml` | pass | Stage 1 smoke passed with torch seeding available. |
| `python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback` | pass | Stage 2 fallback data-pipeline smoke passed. |
| `python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback` | pass | Stage 3 environment smoke passed with 91 decision events and finite invariants. |
| `python -m experiments.smoke_test_experiments` | pass | Stage 8 smoke passed with five baseline successes and missing learned checkpoint skipped. |
| `python -m experiments.smoke_test_mappo_async` | pass | Stage 9 smoke collected assignment=60, truck=7, bus=24, station=1 transitions. |
| `python -m experiments.smoke_test_reward_model` | pass | Stage 5 offline reward-model smoke passed; warning states the fixture is not final reward-model quality. |
| `python -m compileall -q .` | pass | No syntax errors in the strict-latest merge tree. |
| `python C:\Users\dyy21\.codex\skills\rlaif-mappo-paper-alignment\scripts\check_traceability.py .` | pass | Traceability check passed. |

## Remaining Deferred Gates

The following gates require a working PyTorch runtime, real preference labels,
valid trained checkpoints, and final benchmark configuration:

- `python -m experiments.train_reward_model --config configs/train_reward_model.yaml --data data/preference/ai_preferences.jsonl`
- `python -m experiments.evaluate_reward_model --config configs/train_reward_model.yaml --checkpoint results/checkpoints/reward_model.pt`
- `python -m experiments.train_assignment_ppo --config configs/train_assignment_ppo.yaml`
- `python -m experiments.train_mappo_async --config configs/train_mappo_async.yaml`
- configured benchmark, ablation, and sensitivity runs

These are not paper-ready results until completed in the intended runtime
environment with valid AI/human preference labels and trained checkpoints.
