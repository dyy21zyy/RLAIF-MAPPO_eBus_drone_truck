# Experiment Log

Use one entry per run. Keep raw artifacts outside this Markdown file and link to
their stable location.

## Entry template

- **Date/time (UTC):**
- **Stage:**
- **Purpose:**
- **Git commit:**
- **Config:**
- **Seed:**
- **Command:**
- **Environment:** Python/package/hardware details
- **Artifact directory:**
- **Status:** passed / failed / interrupted
- **Key metrics:**
- **Observations:**
- **Follow-up:**

## Stage 1 baseline

- **Purpose:** Validate the repository foundation without internet access.
- **Config:** `configs/shanghai_small.yaml`
- **Command:** `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml`
- **Expected status:** passed
- **Expected artifact:** `logs/smoke_test_project.log`

## 2026-06-09 Stage 1 verification

- **Date/time (UTC):** 2026-06-09
- **Stage:** 1
- **Purpose:** Verify config loading, folders, logging, seed control, and placeholders.
- **Config:** `configs/shanghai_small.yaml`
- **Seed:** 42
- **Command:** `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml`
- **Status:** passed
- **Result:** All required Stage 1 checks completed without network access.

## 2026-06-09 Stage 1 auto-remediation

- **Date/time (UTC):** 2026-06-09
- **Stage:** 1
- **Purpose:** Fix parser-independent empty-configuration validation and add
  automated Stage 1 regression coverage.
- **Config:** `configs/shanghai_small.yaml`
- **Seed:** 42
- **Commands and results:**
  - `python -m pytest -q` — **FAIL** in remediation round 1: 3 tests used
    parser-specific error-message expectations; 12 tests passed.
  - `python -m pytest -q` — **PASS** after narrowing assertions to the public
    `ConfigError` contract: 15 passed.
  - `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml`
    — **PASS**.
  - `python -m compileall -q .` — **PASS**.
  - `git diff --check` — **PASS**.
  - `git status --short --branch` — **PASS** with only expected remediation
    changes before commit.
- **Status:** passed
- **Result:** Empty and empty-root configurations are rejected consistently;
  Stage 1 utilities, placeholders, logging, seeding, runtime folders, and the
  offline smoke path now have automated regression coverage.
