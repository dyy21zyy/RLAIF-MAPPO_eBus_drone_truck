# Paper-Code Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the RLAIF-MAPPO repository with the confirmed paper contract and record traceable validation evidence.

**Architecture:** Treat the manuscript as a requirements contract, map each claim to code and tests, and preserve a strict distinction between code gates and runtime experiment validation. Keep provenance and no-fabrication guardrails central to data, RLAIF, and experiment work.

**Tech Stack:** Python 3.10+, pytest, dependency-light smoke tests, optional PyTorch runtime, Markdown traceability records.

## Global Constraints

- Do not fabricate real data, labels, rewards, checkpoints, or experiment results.
- Keep raw real transit data and generated runtime artifacts out of Git.
- Distinguish code gates from PyTorch runtime gates.
- Use test-first changes for behavior fixes.
- Preserve current Stage 8 framework status unless final runtime experiments are actually run.

---

### Task 1: Preflight and Baseline

**Files:**
- Modify: `docs/paper_code_alignment/validation_report.md`

**Interfaces:**
- Consumes: current repository checkout
- Produces: baseline command evidence

- [x] Record branch, Python version, PyTorch runtime availability, and dirty status.
- [x] Run `python -m pytest -q` and record the initial broken-PyTorch failure.
- [x] Run `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml`.
- [x] Run `git diff --check`.

### Task 2: Runtime Availability Hardening

**Files:**
- Modify: `rlaif/torch_runtime.py`
- Modify: `utils/seeding.py`
- Modify: `experiments/*assignment_ppo*.py`, `experiments/*mappo*.py`, and Stage 5 reward-model CLIs
- Modify: PyTorch-dependent tests

**Interfaces:**
- Produces: `is_torch_runtime_available() -> bool`

- [x] Write failing tests for broken PyTorch handling.
- [x] Implement subprocess-based PyTorch runtime probing with UTF-8 replacement decoding.
- [x] Route tests and smoke commands through the shared helper.
- [x] Verify focused tests pass.

### Task 3: Traceability Records

**Files:**
- Create: `docs/paper_code_alignment/requirements_traceability.md`
- Create: `docs/paper_code_alignment/decision_log.md`
- Create: `docs/paper_code_alignment/validation_report.md`
- Create: `tests/test_paper_code_alignment_docs.py`

**Interfaces:**
- Produces: requirement IDs `REQ-DATA-*`, `REQ-MDP-*`, `REQ-RLAIF-*`, `REQ-MAPPO-*`, `REQ-EXP-*`

- [x] Write failing documentation-presence test.
- [x] Add traceability table with satisfied, partial, and blocked statuses.
- [x] Add validation report with command evidence.
- [x] Add decision log for unresolved manuscript choices.

### Task 4: Reusable Alignment Skill

**Files:**
- Create/update: `C:\Users\dyy21\.codex\skills\rlaif-mappo-paper-alignment\SKILL.md`
- Create/update: `references/*.md`
- Create/update: `scripts/check_traceability.py`
- Create/update: `scripts/run_alignment_gates.ps1`

**Interfaces:**
- Produces: reusable Codex skill for future paper-code alignment work

- [x] Initialize the skill with skill-creator.
- [x] Replace template instructions with the alignment workflow.
- [x] Add reference files and deterministic helper scripts.
- [x] Validate the skill folder.

### Task 5: Final Gates

**Files:**
- Modify: `docs/paper_code_alignment/validation_report.md`

**Interfaces:**
- Consumes: completed code and docs
- Produces: final verification record

- [x] Run `python -m pytest -q`.
- [x] Run smoke commands that do not require PyTorch.
- [x] Run `python -m compileall -q .` from the short `work/rlaif-align` junction.
- [x] Run `git diff --check`.
- [x] Audit `git status --short` for ignored runtime artifacts.
