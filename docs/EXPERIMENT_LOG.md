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

## 2026-06-09 Stage 2 Shanghai data pipeline

- **Date/time (UTC):** 2026-06-09
- **Stage:** 2
- **Purpose:** Build and gate-review the reproducible Shanghai data pipeline.
- **Config:** `configs/shanghai_small.yaml`
- **Seed:** 42
- **Mode:** deterministic offline fallback
- **Artifact directory:** `data/processed/shanghai_yangpu/` (generated and Git-ignored)
- **Commands run:**
  - `python -m experiments.smoke_test_project --config configs/shanghai_small.yaml`
    — **PASS** before Stage 2 implementation; Stage 1 gate remained satisfied.
  - `python -m pytest -q tests/test_stage1_foundation.py` — **PASS**, 15 tests.
  - `python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback`
    — initial gate run **FAIL** because dependency-free config loading did not
    recognize the JSON-compatible `instance.yaml`; remediation rerun **PASS**.
  - `python -m pytest -q` — **PASS**, 22 tests after remediation.
  - `python -m compileall -q .` — **PASS**.
  - `git diff --check` — **PASS**.
  - `git status --short --branch` — reviewed before commit.
- **Generated files:** `road_graph.graphml`, `road_nodes.csv`, `road_edges.csv`,
  `depot.csv`, `bus_stops.csv`, `bus_trips.csv`, `bus_stop_times.csv`,
  `bus_timetable.json`, `integrated_stations.csv`, `parcels.csv`,
  `truck_distance_matrix.npy`, `truck_travel_time_matrix.npy`,
  `drone_distance_matrix.npy`, `instance.yaml`, and `instance.json`.
- **Generated counts:** 25 road nodes, 80 directed road edges, 20 bus stops,
  37 bus trips, 740 stop-time records, 6 integrated stations, and 60 parcels.
- **Matrix shapes:** truck distance/time matrices `68 x 68`; drone distance
  matrix `6 x 60`.
- **Known limitations:** fallback roads are a coarse synthetic grid; the fallback
  route is a geometric corridor rather than an official Shanghai transit route;
  synthetic parcels are not calibrated to private demand; full mode depends on
  optional `osmnx`, public OSM availability, and network access; no simulator or
  learning behavior is included.
- **Rerun command:**
  `python -m data_pipeline.build_instance --config configs/shanghai_small.yaml --fallback`
- **Status:** passed after one auto-remediation round.
