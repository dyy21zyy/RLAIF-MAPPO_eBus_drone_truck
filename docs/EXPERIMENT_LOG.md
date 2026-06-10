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

## Stage 3 gate review before Stage 4 — 2026-06-10

- **Initial result:** failed because the required
  `experiments.smoke_test_env` module did not exist. Inspection also found the
  required reward-component/metrics interface absent and station power enforced
  as a hard charging mask rather than a soft penalty.
- **Remediation:** added the stable fallback gate CLI, exposed reward components
  and sanity metrics, converted station overload to a soft penalty, and added
  station-power/resource regression tests.
- **Final fallback metrics:** 79 decisions (60 assignment, 19 bus), 7 delivered,
  53 undelivered, 0 drone deliveries under the first-feasible policy, reward
  `-6561.086226965998`, 0 corrected actions, and no NaN or negative resources.
- **Result:** Stage 3 passed and Stage 4 was authorized.

## Stage 4 RLAIF data gate — 2026-06-10

- **50-episode output:** 3,000 assignment states and 3,150 pairwise prompts.
- **Offline labels:** 0; no API was used and no rule labels were generated.
- **Smoke output:** 60 states, 63 prompts, 0 labels, and 0 failures.
- **Coverage:** assignment/candidate schema, JSON-only prompt contract, pair
  selection, malformed AI output, low confidence, replay validation, and offline
  no-label behavior.
- **Scope:** no reward-model training, PPO, MAPPO, baselines, or sensitivity
  experiments were added.

## Stage 5 — learned assignment reward model

Implemented the offline pairwise dataset join/filter, deterministic tiny-data
split, training-only feature normalization, action-embedding MLP, Bradley–Terry
loss, early-stopped training, checkpoint round-trip, evaluation CLI, and an
API-free temporary-fixture smoke test. The local ten-label replay dataset is
intended only for pipeline validation; reported one-example validation/test
accuracies must not be interpreted as model-quality estimates. Runtime outputs
are written below `results/` and remain ignored by Git.

### 2026-06-10 Stage 5 Code Gate

The Stage 5 gate is now split into a dependency-light **Code Gate** and a PyTorch
**Runtime Gate**. The Code Gate passed in the current environment: Stage 5 source
and interfaces exist, dataset/missing-label checks run without PyTorch, runtime
commands report a clear installation requirement, PyTorch-dependent tests skip,
and generated labels/checkpoints/results remain ignored. No model was trained and
no training success is claimed.

The Runtime Gate is deferred to the operator's local/AutoDL PyTorch environment:

```bash
python -m experiments.smoke_test_reward_model
python -m experiments.train_reward_model --config configs/train_reward_model.yaml --data data/preference/ai_preferences.jsonl
python -m experiments.evaluate_reward_model --config configs/train_reward_model.yaml --checkpoint results/checkpoints/reward_model.pt
python -m pytest -q
```

Stage 6 code implementation may proceed with `rlaif_enabled=false` smoke coverage.
`rlaif_enabled=true` and final RLAIF-enabled PPO/MAPPO experiments require a valid
trained `reward_model.pt` from the deferred Runtime Gate.
