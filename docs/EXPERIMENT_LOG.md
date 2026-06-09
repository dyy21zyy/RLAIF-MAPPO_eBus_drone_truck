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
