# Paper-Code Alignment Validation Report

## Current Status

Implementation is complete through the Stage 9 four-agent asynchronous MAPPO
code gate for the 2026-07-20 Solution Method alignment. No preference labels, learned rewards, checkpoints, or final results are fabricated.

## Runtime Boundary

PyTorch runtime gates require a working PyTorch environment and valid trained
checkpoints. Missing checkpoints are recorded as skipped or deferred, not
substituted with heuristics.

## Verification Log

| Command | Status | Evidence |
| --- | --- | --- |
| `python -m pytest tests/test_paper_alignment_traceability.py -q` | pass | Initial traceability docs were created and validated. |
| `python -m pytest tests/test_four_agent_environment.py tests/test_stage3_environment.py -q` | pass | Four active agent ids and Stage 3 regressions passed during Task 3. |
| `python -m pytest tests/test_mappo_buffer.py -q` | pass | Four-agent buffer and event-time discounting passed during Task 4. |
| `python -m pytest tests/test_mappo_networks.py -q` | pass | Candidate-scoring actor and actor registry passed during Task 5. |
| `python -m pytest tests/test_mappo_async.py tests/test_mappo_networks.py tests/test_mappo_buffer.py -q` | pass | Stage 9 trainer smoke contract and MAPPO unit tests passed during Task 6. |
| `python -m pytest tests/test_paper_alignment_traceability.py tests/test_four_agent_candidate_schema.py tests/test_four_agent_environment.py tests/test_mappo_buffer.py tests/test_mappo_networks.py tests/test_mappo_async.py -q` | pass | Final focused Stage 9 gate: 24 passed. |
| `python -m pytest -q` | pass | Final full suite: 146 passed, 3 skipped, 1 known reward-model smoke warning. |
| `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml` | pass | Stage 1 smoke passed. |
| `python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback` | pass | Stage 2 fallback data-pipeline smoke passed. |
| `python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback` | pass | Stage 3 event-driven environment smoke passed. |
| `python -m experiments.smoke_test_experiments` | pass | Stage 8 smoke passed with five baseline successes and missing learned checkpoint skipped. |
| `python -m experiments.smoke_test_mappo_async` | pass | Stage 9 smoke collected assignment=60, truck=7, bus=24, station=1 transitions. |
| `python -m compileall -q .` | pass | No syntax errors. |
| `git diff --check` | pass | No whitespace errors. |
