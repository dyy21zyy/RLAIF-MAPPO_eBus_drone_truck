# Four-Agent RLAIF-MAPPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete four-agent event-driven RLAIF-MAPPO code path required by the confirmed 2026-07-20 Solution Method manuscript.

**Architecture:** Extend the existing Stage 7 MAPPO implementation into a Stage 9 code path instead of replacing the repo. `DynamicDeliveryEnv` remains the event-driven simulator, but it exposes real assignment, truck, bus, and station decisions with candidate actions, candidate features, and masks. Training uses a per-agent actor registry, candidate-scoring masked actors, a shared centralized critic, an asynchronous event buffer, and strict checkpoint-backed RLAIF reward boundaries.

**Tech Stack:** Python 3, NumPy, PyTorch when available, pytest, existing repo config/data pipeline utilities.

## Global Constraints

- Treat the confirmed 2026-07-20 manuscript as the behavior source of truth for this pass.
- Do not fabricate real transit data, preference labels, learned rewards, checkpoints, benchmark results, ablation results, or sensitivity results.
- Keep runtime artifacts under ignored output paths and out of Git.
- Use TDD for behavior changes: write a focused failing test, verify RED, implement the minimal passing code, verify GREEN, then refactor.
- Preserve the existing staged smoke gates where possible; add Stage 9 behavior without silently changing previous experiment claims.
- Default `rlaif.enabled` remains false for code-gate smoke tests; enabled RLAIF requires a valid Stage 5 checkpoint and must fail clearly otherwise.
- Smoke tests validate interfaces and invariants only; they are not final paper experiment evidence.

---

## File Structure

- Create `docs/paper_code_alignment/requirements_traceability.md`: requirement IDs, paper claims, code areas, tests, status, and evidence.
- Create `docs/paper_code_alignment/decision_log.md`: explicit decision that the current pass supersedes the earlier two-agent Stage 7 MAPPO boundary.
- Create `docs/paper_code_alignment/validation_report.md`: commands run, pass/fail/skip state, and runtime gates deferred because no final trained checkpoint is produced here.
- Create `envs/decision_schema.py`: small dataclasses and helpers for candidate actions and decision observations.
- Modify `envs/state_builder.py`: add feature names and candidate builders for truck, bus loading, bus charging, and station operation while retaining existing assignment feature helpers.
- Modify `envs/delivery_env.py`: add four-agent decision events, task queues, station battery-operation state, observation payload extensions, and action appliers.
- Modify `training/mappo_async.py`: generalize valid agent/event pairs and RLAIF reward scope.
- Modify `training/mappo_buffer.py`: support four agent ids, event-time discounting, and transition candidate metadata.
- Modify `training/mappo_networks.py`: add candidate-scoring actor and four named actor wrappers.
- Modify `training/mappo_trainer.py`: replace hard-coded assignment/bus branches with an actor registry, per-agent optimizers, per-agent metrics, and Stage 9 checkpoint metadata.
- Modify `experiments/smoke_test_mappo_async.py`: assert all four agent types are collected in the smoke path when PyTorch is available.
- Modify `docs/MDP_SPECIFICATION.md`, `docs/RLAIF_WORKFLOW.md`, `docs/EXPERIMENTS.md`, and `docs/PITFALLS.md`: describe the Stage 9 four-agent code gate and preserve the final-runtime boundary.
- Create or modify focused tests under `tests/` as listed below.

---

### Task 1: Paper-Code Alignment Traceability

**Files:**
- Create: `docs/paper_code_alignment/requirements_traceability.md`
- Create: `docs/paper_code_alignment/decision_log.md`
- Create: `docs/paper_code_alignment/validation_report.md`
- Create: `tests/test_paper_alignment_traceability.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-20-four-agent-rlaif-mappo-design.md`
- Produces: Markdown files that later tasks update with evidence lines for `REQ-MDP-*`, `REQ-MAPPO-*`, `REQ-RLAIF-*`, and `REQ-DOC-*`.

- [ ] **Step 1: Write the failing traceability test**

```python
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_four_agent_traceability_records_core_requirements():
    trace = _text("docs/paper_code_alignment/requirements_traceability.md")
    for requirement in (
        "REQ-MDP-FOUR-AGENT",
        "REQ-MDP-CANDIDATES",
        "REQ-MAPPO-ACTORS",
        "REQ-MAPPO-CANDIDATE-POLICY",
        "REQ-MAPPO-BUFFER",
        "REQ-RLAIF-SCOPE",
    ):
        assert requirement in trace
    assert "assignment, truck, bus, and station" in trace
    assert "not final experiment evidence" in trace


def test_decision_log_supersedes_two_agent_stage7_boundary():
    decision_log = _text("docs/paper_code_alignment/decision_log.md")
    assert "2026-07-20" in decision_log
    assert "supersedes the previous Stage 7 two-agent boundary" in decision_log
    assert "four-agent" in decision_log


def test_validation_report_keeps_no_fabrication_boundary():
    report = _text("docs/paper_code_alignment/validation_report.md")
    assert "No preference labels, learned rewards, checkpoints, or final results are fabricated" in report
    assert "PyTorch runtime gates" in report
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_paper_alignment_traceability.py -q`

Expected: FAIL because `docs/paper_code_alignment/requirements_traceability.md` does not exist.

- [ ] **Step 3: Write minimal alignment docs**

Create `requirements_traceability.md` with this table header and required rows:

```markdown
# Paper-Code Alignment Traceability

| ID | Paper claim | Required behavior | Code area | Test | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| REQ-MDP-FOUR-AGENT | The method trains assignment, truck, bus, and station agents over the asynchronous event stream. | Expose real decision events for assignment, truck, bus, and station without inactive-agent padding. | `envs/delivery_env.py`; `envs/state_builder.py` | `tests/test_four_agent_environment.py` | planned | Design committed in `docs/superpowers/specs/2026-07-20-four-agent-rlaif-mappo-design.md`. |
| REQ-MDP-CANDIDATES | Candidate-action encoding and masks represent operational constraints. | Every decision observation includes candidate descriptors, finite candidate features, and hard masks. | `envs/decision_schema.py`; `envs/state_builder.py` | `tests/test_four_agent_candidate_schema.py` | planned | To be verified by TDD task. |
| REQ-MAPPO-ACTORS | Heterogeneous event-specific actors train under a centralized critic. | Provide actor registry entries for assignment, truck, bus, and station. | `training/mappo_networks.py`; `training/mappo_trainer.py` | `tests/test_mappo_networks.py`; `tests/test_mappo_async.py` | planned | To be verified by TDD task. |
| REQ-MAPPO-CANDIDATE-POLICY | The actor scores candidate actions and applies masks before softmax. | Candidate-scoring actors give infeasible actions zero probability. | `training/mappo_networks.py` | `tests/test_mappo_networks.py` | planned | To be verified by TDD task. |
| REQ-MAPPO-BUFFER | Rollouts store only activated agents and use event-time discounting. | Buffer supports four agent ids, no inactive rows, and time-delta GAE. | `training/mappo_buffer.py` | `tests/test_mappo_buffer.py` | planned | To be verified by TDD task. |
| REQ-RLAIF-SCOPE | AI feedback ranks feasible alternatives, while physical feasibility remains masked. | Disabled RLAIF loads no checkpoint; enabled RLAIF requires a valid checkpoint; no rules create labels or rewards. | `rlaif/`; `training/reward_model_wrapper.py`; `training/mappo_async.py` | `tests/test_reward_model_wrapper.py`; `tests/test_mappo_async.py` | planned | Existing Stage 5/7 guardrails remain active. |

Smoke-test metrics are code-gate evidence only and are not final experiment evidence.
```

Create `decision_log.md`:

```markdown
# Paper-Code Alignment Decision Log

## 2026-07-20 Four-Agent Solution Method Scope

The user confirmed that this implementation pass must follow the complete
four-agent document alignment. This supersedes the previous Stage 7 two-agent
boundary where MAPPO controlled assignment and bus charging only.

The current pass treats the supplied manuscript as source of truth for code
alignment. It must not fabricate preference labels, learned rewards,
checkpoints, real data, benchmark results, ablation results, or sensitivity
results.
```

Create `validation_report.md`:

```markdown
# Paper-Code Alignment Validation Report

## Current Status

Implementation is in progress for the 2026-07-20 four-agent Solution Method
alignment. No preference labels, learned rewards, checkpoints, or final results
are fabricated.

## Runtime Boundary

PyTorch runtime gates require a working PyTorch environment and valid trained
checkpoints. Missing checkpoints are recorded as skipped or deferred, not
substituted with heuristics.
```

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_paper_alignment_traceability.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add docs/paper_code_alignment tests/test_paper_alignment_traceability.py
git commit -m "docs: add four-agent paper alignment traceability"
```

---

### Task 2: Shared Decision And Candidate Schema

**Files:**
- Create: `envs/decision_schema.py`
- Create: `tests/test_four_agent_candidate_schema.py`

**Interfaces:**
- Consumes: no task-specific production interfaces.
- Produces:
  - `ActionCandidate(action_id: int, action_type: str, entity_id: str, description: str, features: dict[str, float], feasible: bool, reasons: tuple[str, ...])`
  - `DecisionSurface(agent_id: str, event_type: str, entity_id: str, features: list[float], feature_names: tuple[str, ...], candidates: list[ActionCandidate])`
  - `DecisionSurface.action_mask() -> list[bool]`
  - `DecisionSurface.candidate_feature_names() -> tuple[str, ...]`
  - `DecisionSurface.candidate_feature_matrix() -> list[list[float]]`
  - `DecisionSurface.candidate_payloads() -> list[dict[str, object]]`

- [ ] **Step 1: Write the failing schema test**

```python
import pytest

from envs.decision_schema import ActionCandidate, DecisionSurface


def test_decision_surface_exports_mask_and_candidate_matrix():
    surface = DecisionSurface(
        agent_id="truck",
        event_type="TRUCK_AVAILABLE",
        entity_id="truck_000",
        features=[0.25, 1.0],
        feature_names=("time_norm", "capacity_norm"),
        candidates=[
            ActionCandidate(
                action_id=0,
                action_type="execute_task",
                entity_id="parcel_001",
                description="direct delivery",
                features={"estimated_time_norm": 0.2, "idle_flag": 0.0},
                feasible=True,
                reasons=(),
            ),
            ActionCandidate(
                action_id=1,
                action_type="idle",
                entity_id="truck_000",
                description="remain idle",
                features={"estimated_time_norm": 0.0, "idle_flag": 1.0},
                feasible=False,
                reasons=("task_available",),
            ),
        ],
    )

    assert surface.action_mask() == [True, False]
    assert surface.candidate_feature_names() == ("estimated_time_norm", "idle_flag")
    assert surface.candidate_feature_matrix() == [[0.2, 0.0], [0.0, 1.0]]
    payload = surface.candidate_payloads()[0]
    assert payload["action_type"] == "execute_task"
    assert payload["features"]["estimated_time_norm"] == 0.2


def test_decision_surface_requires_one_feasible_candidate():
    with pytest.raises(ValueError, match="at least one feasible"):
        DecisionSurface(
            agent_id="station",
            event_type="STATION_OPERATION",
            entity_id="station_001",
            features=[0.0],
            feature_names=("time_norm",),
            candidates=[
                ActionCandidate(0, "dispatch", "parcel_001", "dispatch", {"x": 1.0}, False, ("no_battery",))
            ],
        )
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_four_agent_candidate_schema.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'envs.decision_schema'`.

- [ ] **Step 3: Implement the minimal schema module**

Create `envs/decision_schema.py`:

```python
"""Shared decision and candidate-action schema for four-agent event control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ActionCandidate:
    action_id: int
    action_type: str
    entity_id: str
    description: str
    features: dict[str, float]
    feasible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action_id < 0:
            raise ValueError("action_id must be non-negative")
        for key, value in self.features.items():
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"Candidate feature {key!r} must be finite")
            object.__setattr__(self, "features", {**self.features, key: numeric})

    def payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "entity_id": self.entity_id,
            "description": self.description,
            "features": dict(self.features),
            "feasible": bool(self.feasible),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DecisionSurface:
    agent_id: str
    event_type: str
    entity_id: str
    features: list[float]
    feature_names: tuple[str, ...]
    candidates: list[ActionCandidate]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.feature_names):
            raise ValueError("features and feature_names must have the same length")
        if not self.candidates:
            raise ValueError("Decision surface requires candidates")
        if not any(candidate.feasible for candidate in self.candidates):
            raise ValueError("Decision surface requires at least one feasible candidate")
        for value in self.features:
            if not isfinite(float(value)):
                raise ValueError("Decision features must be finite")

    def action_mask(self) -> list[bool]:
        return [bool(candidate.feasible) for candidate in self.candidates]

    def candidate_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for candidate in self.candidates:
            for key in candidate.features:
                if key not in names:
                    names.append(key)
        return tuple(names)

    def candidate_feature_matrix(self) -> list[list[float]]:
        names = self.candidate_feature_names()
        return [
            [float(candidate.features.get(name, 0.0)) for name in names]
            for candidate in self.candidates
        ]

    def candidate_payloads(self) -> list[dict[str, Any]]:
        return [candidate.payload() for candidate in self.candidates]
```

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_four_agent_candidate_schema.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add envs/decision_schema.py tests/test_four_agent_candidate_schema.py
git commit -m "feat: add four-agent decision schema"
```

---

### Task 3: Four-Agent Environment Decision Exposure

**Files:**
- Modify: `envs/delivery_env.py`
- Modify: `envs/state_builder.py`
- Modify: `tests/test_stage3_environment.py`
- Create: `tests/test_four_agent_environment.py`

**Interfaces:**
- Consumes: `envs.decision_schema.ActionCandidate`, `envs.decision_schema.DecisionSurface`.
- Produces:
  - Observation keys `candidate_actions`, `candidate_features`, and `candidate_feature_names`.
  - Agent ids `assignment`, `truck`, `bus`, `station`, `terminal`.
  - Event types `PARCEL_RELEASE`, `TRUCK_AVAILABLE`, `BUS_DEPARTURE`, `BUS_ARRIVAL`, `STATION_OPERATION`, `TERMINAL`.

- [ ] **Step 1: Write the failing four-agent exposure tests**

```python
from pathlib import Path

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, first_feasible_policy

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


def make_env(tmp_path):
    instance = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    return DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")


def collect_agents(env, limit=500):
    observation, _ = env.reset(seed=11)
    seen = []
    while observation["agent_id"] != "terminal" and len(seen) < limit:
        seen.append((observation["agent_id"], observation["event_type"]))
        assert "candidate_actions" in observation
        assert "candidate_features" in observation
        assert "candidate_feature_names" in observation
        assert len(observation["candidate_actions"]) == len(observation["action_mask"])
        assert len(observation["candidate_features"]) == len(observation["action_mask"])
        assert any(observation["action_mask"])
        observation, *_ = env.step(first_feasible_policy(observation))
        assert env.check_invariants() == []
    return seen


def test_episode_exposes_assignment_truck_bus_and_station_decisions(tmp_path):
    env = make_env(tmp_path)
    seen = collect_agents(env)
    agent_ids = {agent for agent, _event in seen}
    assert {"assignment", "truck", "bus", "station"} <= agent_ids


def test_four_agent_event_types_are_operational_not_dummy(tmp_path):
    env = make_env(tmp_path)
    seen = collect_agents(env)
    pairs = set(seen)
    assert ("assignment", "PARCEL_RELEASE") in pairs
    assert ("truck", "TRUCK_AVAILABLE") in pairs
    assert ("bus", "BUS_DEPARTURE") in pairs
    assert ("bus", "BUS_ARRIVAL") in pairs
    assert ("station", "STATION_OPERATION") in pairs
    assert all(agent != "inactive" for agent, _event in seen)
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_four_agent_environment.py -q`

Expected: FAIL because current episodes only expose `assignment` and `bus`, and observations lack candidate payload keys.

- [ ] **Step 3: Extend event constants and reset state**

In `envs/delivery_env.py`, add event priorities:

```python
EVENT_PRIORITY = {
    "battery_ready": 0,
    "drone_return": 1,
    "station_operation": 2,
    "drone_dispatch": 3,
    "parcel_delivery": 4,
    "parcel_station_arrival": 5,
    "bus_departure": 6,
    "bus_arrival": 7,
    "truck_available": 8,
    "parcel_release": 9,
}
```

In `reset()`, initialize queues:

```python
self.decision_counts = {"assignment": 0, "truck": 0, "bus": 0, "station": 0}
self.pending_truck_tasks: list[dict[str, Any]] = []
self.waiting_station_parcels: dict[str, list[str]] = {}
self.bus_terminal_ready: dict[str, list[str]] = {}
self.bus_loaded_parcels: dict[tuple[str, str], list[str]] = {}
```

Push `bus_departure` events for freight-enabled trips at their first stop:

```python
if self._as_bool(self.trip_rows[trip_id]["freight_allowed"]):
    self._push(float(rows[0]["departure_time"]), "bus_departure", {"trip_id": trip_id})
```

- [ ] **Step 4: Add decision-surface builders**

In `envs/state_builder.py`, add feature-name constants and simple finite builders:

```python
TRUCK_FEATURE_NAMES = ("time_norm", "available_now", "pending_task_count_norm", "capacity_norm")
BUS_LOADING_FEATURE_NAMES = ("time_norm", "ready_parcel_count_norm", "freight_load_norm", "capacity_remaining_norm")
BUS_CHARGING_FEATURE_NAMES = ("time_norm", "soc_norm", "delay_norm", "locker_load_norm", "full_batteries_norm", "freight_load_norm")
STATION_FEATURE_NAMES = ("time_norm", "waiting_parcels_norm", "locker_load_norm", "idle_drones_norm", "full_batteries_norm", "power_margin_norm")
COMMON_CANDIDATE_FEATURE_NAMES = ("action_type_id", "estimated_time_norm", "estimated_lateness_norm", "capacity_after_norm", "resource_margin_norm", "idle_flag")
```

Add builder functions with these exact signatures:

- `build_truck_decision_surface(env: Any, truck: Any) -> DecisionSurface`
- `build_bus_loading_decision_surface(env: Any, trip_id: str) -> DecisionSurface`
- `build_bus_charging_decision_surface(env: Any, event: Any) -> DecisionSurface`
- `build_station_decision_surface(env: Any, station_id: str) -> DecisionSurface`

Each builder returns at least one feasible candidate. Candidate index `0` is the productive action when one exists and an idle action when no productive action is feasible.

- [ ] **Step 5: Route events to four decision surfaces**

In `_advance()`, set `current_decision` for new events:

```python
elif event.kind == "truck_available":
    truck = self.trucks[event.payload["truck_index"]]
    surface = build_truck_decision_surface(self, truck)
    self.current_decision = Decision("truck", event, surface.action_mask())
elif event.kind == "bus_departure":
    surface = build_bus_loading_decision_surface(self, event.payload["trip_id"])
    self.current_decision = Decision("bus", event, surface.action_mask())
elif event.kind == "station_operation":
    surface = build_station_decision_surface(self, event.payload["station_id"])
    self.current_decision = Decision("station", event, surface.action_mask())
```

In `_observation()`, return the `DecisionSurface` payload for all agents and map the event names exactly to the test strings.

- [ ] **Step 6: Add minimal action application**

In `step()`, dispatch actions by agent:

```python
elif decision.agent == "truck":
    reward += self._apply_truck_action(decision.event, selected)
elif decision.agent == "bus":
    if decision.event.kind == "bus_departure":
        reward += self._apply_bus_loading_action(decision.event, selected)
    else:
        reward += self._apply_bus_action(decision.event, selected)
elif decision.agent == "station":
    reward += self._apply_station_action(decision.event, selected)
```

Use existing `_record_truck_trip()`, `_handle_bus_arrival()`, and `_dispatch_drone()` behavior where possible, but move the policy choice to the relevant agent event.

- [ ] **Step 7: Run test to verify GREEN**

Run: `python -m pytest tests/test_four_agent_environment.py -q`

Expected: PASS.

- [ ] **Step 8: Run Stage 3 regression tests**

Run: `python -m pytest tests/test_stage3_environment.py -q`

Expected: PASS after updating legacy assertions to accept four-agent metrics and the extended observation payload.

- [ ] **Step 9: Commit**

```powershell
git add envs/delivery_env.py envs/state_builder.py tests/test_stage3_environment.py tests/test_four_agent_environment.py
git commit -m "feat: expose four-agent event decisions"
```

---

### Task 4: Four-Agent MAPPO Buffer And Event-Time Discounting

**Files:**
- Modify: `training/mappo_buffer.py`
- Modify: `tests/test_mappo_buffer.py`

**Interfaces:**
- Consumes: transition fields already used by Stage 7 plus `candidate_features` and `candidate_feature_names`.
- Produces:
  - `VALID_AGENTS = {"assignment", "truck", "bus", "station"}`
  - `AsyncTransition.candidate_features: list[list[float]]`
  - `AsyncTransition.candidate_feature_names: tuple[str, ...]`
  - `AsyncMAPPOBuffer.compute_returns_and_advantages(gamma, gae_lambda, reference_time_unit=1.0)`

- [ ] **Step 1: Write failing buffer tests**

Add to `tests/test_mappo_buffer.py`:

```python
def test_buffer_accepts_four_agent_types():
    buffer = AsyncMAPPOBuffer()
    for agent, event in (
        ("assignment", "PARCEL_RELEASE"),
        ("truck", "TRUCK_AVAILABLE"),
        ("bus", "BUS_DEPARTURE"),
        ("station", "STATION_OPERATION"),
    ):
        buffer.append(transition(agent=agent, event=event))
    assert [item.agent_id for item in buffer.transitions] == ["assignment", "truck", "bus", "station"]


def test_event_time_discount_uses_elapsed_minutes():
    buffer = AsyncMAPPOBuffer()
    first = transition(agent="assignment", event="PARCEL_RELEASE", episode=0)
    second = transition(agent="truck", event="TRUCK_AVAILABLE", episode=0)
    first.event_time = 0.0
    second.event_time = 10.0
    first.reward = 0.0
    first.value = 0.0
    second.reward = 1.0
    second.value = 0.0
    second.done = True
    buffer.append(first)
    buffer.append(second)
    returns, _advantages = buffer.compute_returns_and_advantages(0.9, 1.0, reference_time_unit=10.0)
    assert returns[0] == pytest.approx(0.9)
    assert returns[1] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_mappo_buffer.py -q`

Expected: FAIL because `truck` and `station` are rejected or because `reference_time_unit` is not accepted.

- [ ] **Step 3: Implement four-agent buffer support**

Update `VALID_AGENTS`, extend `AsyncTransition`, and compute next discount as:

```python
elapsed = max(0.0, float(next_item.event_time) - float(item.event_time)) if same_episode else float(reference_time_unit)
discount = float(gamma) ** (elapsed / max(float(reference_time_unit), 1e-8))
delta = item.reward + discount * nonterminal * next_value - item.value
gae = delta + discount * float(gae_lambda) * nonterminal * gae
```

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_mappo_buffer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add training/mappo_buffer.py tests/test_mappo_buffer.py
git commit -m "feat: support four-agent event-time mappo buffer"
```

---

### Task 5: Candidate-Scoring Actors And Actor Registry

**Files:**
- Modify: `training/mappo_networks.py`
- Modify: `tests/test_mappo_networks.py`

**Interfaces:**
- Consumes: candidate feature matrices from observations and `AsyncTransition`.
- Produces:
  - `CandidateScoringActor(obs_dim: int, candidate_feature_dim: int, hidden_dims: Sequence[int])`
  - `FourAgentActors` or `build_actor_registry(specs: dict[str, tuple[int, int]], hidden_dims: dict[str, Sequence[int]])`
  - wrappers or registry entries for `assignment`, `truck`, `bus`, and `station`.

- [ ] **Step 1: Write failing actor tests**

Add to `tests/test_mappo_networks.py`:

```python
from training.mappo_networks import CandidateScoringActor, build_actor_registry


def test_candidate_scoring_actor_respects_variable_candidate_mask():
    actor = CandidateScoringActor(obs_dim=3, candidate_feature_dim=2, hidden_dims=[8])
    observation = torch.zeros(3)
    candidates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([False, True, False])
    assert all(actor.act(observation, candidates, mask)[0] == 1 for _ in range(20))


def test_actor_registry_contains_four_agent_types():
    registry = build_actor_registry(
        {
            "assignment": (4, 3),
            "truck": (5, 3),
            "bus": (6, 3),
            "station": (7, 3),
        },
        {"default": [8]},
    )
    assert set(registry) == {"assignment", "truck", "bus", "station"}
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_mappo_networks.py -q`

Expected: FAIL because `CandidateScoringActor` is missing.

- [ ] **Step 3: Implement candidate-scoring actor**

Add a class that concatenates each candidate row with the repeated observation and scores candidates:

```python
class CandidateScoringActor(nn.Module):
    def __init__(self, obs_dim: int, candidate_feature_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.scorer = _mlp(self.obs_dim + self.candidate_feature_dim, tuple(hidden_dims), 1)

    def distribution(self, observations: Tensor, candidate_features: Tensor, action_masks: Tensor) -> Categorical:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if candidate_features.ndim == 2:
            candidate_features = candidate_features.unsqueeze(0)
        if action_masks.ndim == 1:
            action_masks = action_masks.unsqueeze(0)
        obs = observations.float().unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        inputs = torch.cat((obs, candidate_features.float()), dim=-1)
        logits = self.scorer(inputs).squeeze(-1)
        masks = action_masks.to(device=logits.device, dtype=torch.bool)
        if (~masks.any(dim=-1)).any():
            raise ValueError("Action mask must contain at least one feasible action")
        return Categorical(logits=logits.masked_fill(~masks, MASKED_LOGIT))

    def act(self, observation, candidate_features, action_mask, *, deterministic: bool = False) -> tuple[int, float]:
        with torch.no_grad():
            distribution = self.distribution(
                torch.as_tensor(observation, dtype=torch.float32),
                torch.as_tensor(candidate_features, dtype=torch.float32),
                torch.as_tensor(action_mask, dtype=torch.bool),
            )
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            return int(action.item()), float(distribution.log_prob(action).item())

    def evaluate_actions(self, observations: Tensor, candidate_features: Tensor, actions: Tensor, action_masks: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(observations, candidate_features, action_masks)
        actions = actions.long().reshape(-1)
        return distribution.log_prob(actions), distribution.entropy()
```

Add `build_actor_registry()` returning `nn.ModuleDict`.

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_mappo_networks.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add training/mappo_networks.py tests/test_mappo_networks.py
git commit -m "feat: add candidate-scoring four-agent actors"
```

---

### Task 6: Four-Agent MAPPO Trainer And Smoke Gate

**Files:**
- Modify: `training/mappo_async.py`
- Modify: `training/mappo_trainer.py`
- Modify: `experiments/smoke_test_mappo_async.py`
- Modify: `tests/test_mappo_async.py`

**Interfaces:**
- Consumes: four-agent observations with `candidate_features`.
- Produces:
  - `validate_decision()` accepts the five operational event pairs.
  - `collect_episode()` uses an actor registry keyed by `agent_id`.
  - `update_mappo()` groups actor updates by four agent ids.
  - Training rows include `assignment_decision_count`, `truck_decision_count`, `bus_decision_count`, and `station_decision_count`.
  - Checkpoints include `stage: 9`, `algorithm: four_agent_asynchronous_mappo`, actor dimensions, actor state dicts, and critic state dict.

- [ ] **Step 1: Write failing MAPPO tests**

Update `tests/test_mappo_async.py`:

```python
def test_agent_event_pairs_include_four_agent_solution_method():
    for agent, event in (
        ("assignment", "PARCEL_RELEASE"),
        ("truck", "TRUCK_AVAILABLE"),
        ("bus", "BUS_DEPARTURE"),
        ("bus", "BUS_ARRIVAL"),
        ("station", "STATION_OPERATION"),
    ):
        validate_decision(agent, event)
    with pytest.raises(ValueError):
        validate_decision("station", "BUS_ARRIVAL")


def test_smoke_contract_collects_four_agent_transitions():
    from experiments.smoke_test_mappo_async import run_smoke_test
    result = run_smoke_test()
    assert result["skipped"] or (
        result["assignment_transitions"] > 0
        and result["truck_transitions"] > 0
        and result["bus_transitions"] > 0
        and result["station_transitions"] > 0
    )
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_mappo_async.py -q`

Expected: FAIL because current validation only accepts assignment and bus arrival.

- [ ] **Step 3: Generalize event validation and reward scope**

In `training/mappo_async.py`, define:

```python
VALID_AGENT_EVENTS = {
    "assignment": {"PARCEL_RELEASE"},
    "truck": {"TRUCK_AVAILABLE"},
    "bus": {"BUS_DEPARTURE", "BUS_ARRIVAL"},
    "station": {"STATION_OPERATION"},
}
RLAIF_AGENT_TYPES = {"assignment"}
```

Keep learned reward assignment-only until multi-agent preference checkpoints exist, and return environment reward for truck, bus, and station rows.

- [ ] **Step 4: Rewrite trainer registry path**

In `training/mappo_trainer.py`, collect dimensions from observed decisions, build actors with `build_actor_registry()`, and select actor by `agent_id`:

```python
actor = actors[agent_id]
action, log_prob = actor.act(local_obs, candidate_features, mask, deterministic=deterministic)
```

Store candidate metadata on `AsyncTransition`. In `update_mappo()`, loop over `actors.items()` and use `item.candidate_features`.

- [ ] **Step 5: Update smoke test**

In `experiments/smoke_test_mappo_async.py`, assert four transition counts and checkpoint metadata:

```python
if checkpoint["stage"] != 9 or checkpoint["algorithm"] != "four_agent_asynchronous_mappo":
    raise AssertionError("Checkpoint round trip failed")
```

- [ ] **Step 6: Run test to verify GREEN**

Run: `python -m pytest tests/test_mappo_async.py tests/test_mappo_networks.py tests/test_mappo_buffer.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add training/mappo_async.py training/mappo_trainer.py experiments/smoke_test_mappo_async.py tests/test_mappo_async.py
git commit -m "feat: train four-agent asynchronous mappo"
```

---

### Task 7: RLAIF And Documentation Guardrails

**Files:**
- Modify: `docs/MDP_SPECIFICATION.md`
- Modify: `docs/RLAIF_WORKFLOW.md`
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/PITFALLS.md`
- Modify: `docs/paper_code_alignment/requirements_traceability.md`
- Modify: `docs/paper_code_alignment/decision_log.md`
- Modify: `docs/paper_code_alignment/validation_report.md`
- Modify: `tests/test_documentation_consistency.py`

**Interfaces:**
- Consumes: Stage 9 behavior from previous tasks.
- Produces: docs that describe the four-agent code gate and preserve no-fabrication boundaries.

- [ ] **Step 1: Write failing documentation tests**

Add to `tests/test_documentation_consistency.py`:

```python
def test_docs_describe_stage9_four_agent_mappo_boundary():
    mdp = _read("docs/MDP_SPECIFICATION.md")
    rlaif = _read("docs/RLAIF_WORKFLOW.md")
    experiments = _read("docs/EXPERIMENTS.md")
    combined = "\n".join((mdp, rlaif, experiments))
    assert "Stage 9 four-agent asynchronous MAPPO" in combined
    assert "assignment, truck, bus, and station" in combined
    assert "candidate actions, candidate features, and action masks" in combined


def test_docs_keep_four_agent_rlaif_no_fabrication_boundary():
    rlaif = _read("docs/RLAIF_WORKFLOW.md")
    assert "No rule score, heuristic preference, objective-feature label, evaluator reason text, or blank template may become a learned reward" in rlaif
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_documentation_consistency.py -q`

Expected: FAIL because Stage 9 wording is absent.

- [ ] **Step 3: Update docs**

Add sections that state:

```markdown
# Stage 9 Four-Agent Asynchronous MAPPO Code Gate

Stage 9 aligns the code surface with the 2026-07-20 Solution Method manuscript.
It exposes assignment, truck, bus, and station decisions in the real event
stream. Each active decision provides candidate actions, candidate features, and
action masks. The trainer stores only activated-agent rows, applies event-time
discounting, and trains event-specific actors with a shared centralized critic.

This is a code gate, not final experiment evidence. No rule score, heuristic
preference, objective-feature label, evaluator reason text, or blank template may
become a learned reward.
```

Update traceability rows from `planned` to `implemented` only for requirements
whose tests pass in the current run. Leave runtime preference-reward expansion
as `blocked-by-user-decision-or-runtime-data` when real labels or checkpoints are
missing.

- [ ] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_documentation_consistency.py tests/test_paper_alignment_traceability.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add docs tests/test_documentation_consistency.py
git commit -m "docs: document four-agent mappo code gate"
```

---

### Task 8: Final Verification And Cleanup

**Files:**
- Modify only files needed to fix failures from this task.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: validated current-state evidence for final reporting.

- [ ] **Step 1: Run focused Stage 9 tests**

Run:

```powershell
python -m pytest tests/test_paper_alignment_traceability.py tests/test_four_agent_candidate_schema.py tests/test_four_agent_environment.py tests/test_mappo_buffer.py tests/test_mappo_networks.py tests/test_mappo_async.py -q
```

Expected: PASS or PyTorch-specific tests cleanly skip when PyTorch is unavailable.

- [ ] **Step 2: Run existing dependency-light tests**

Run:

```powershell
python -m pytest -q
```

Expected: PASS with only known smoke/runtime warnings that do not claim final results.

- [ ] **Step 3: Run smoke commands**

Run:

```powershell
python -m experiments.smoke_test_project --config configs/shanghai_small.yaml
python -m experiments.smoke_test_data_pipeline --config configs/shanghai_small.yaml --fallback
python -m experiments.smoke_test_env --config configs/shanghai_small.yaml --fallback
python -m experiments.smoke_test_experiments
python -m experiments.smoke_test_mappo_async
```

Expected: PASS, or `smoke_test_mappo_async` returns a clean PyTorch-unavailable skip.

- [ ] **Step 4: Run syntax and diff gates**

Run:

```powershell
python -m compileall -q .
git diff --check
git status --short
```

Expected: `compileall` and `git diff --check` pass. `git status --short` shows only intentional uncommitted changes before the final commit.

- [ ] **Step 5: Update validation report**

Append exact command results to `docs/paper_code_alignment/validation_report.md`:

```markdown
## Verification Run

| Command | Status | Evidence |
| --- | --- | --- |
| `python -m pytest -q` | pass | Test output from current run. |
| `python -m experiments.smoke_test_mappo_async` | pass-or-skip | Four-agent transition counts when PyTorch is available; clean skip otherwise. |
| `python -m compileall -q .` | pass | No syntax errors. |
| `git diff --check` | pass | No whitespace errors. |
```

- [ ] **Step 6: Commit final validation**

```powershell
git add docs/paper_code_alignment/validation_report.md
git commit -m "test: record four-agent alignment validation"
```

- [ ] **Step 7: Final audit**

Run:

```powershell
git log --oneline -5
git status --short
```

Expected: recent commits include the design, traceability, implementation, docs, and validation commits. Working tree is clean unless the user requested uncommitted changes.

---

## Self-Review

- Spec coverage: Tasks 1 and 7 cover REQ-DOC-TRACE and no-fabrication docs; Tasks 2 and 3 cover REQ-MDP-FOUR-AGENT, REQ-MDP-CANDIDATES, and REQ-MDP-TRANSITIONS; Tasks 4 through 6 cover REQ-MAPPO-ACTORS, REQ-MAPPO-CANDIDATE-POLICY, REQ-MAPPO-BUFFER, and REQ-RLAIF-SCOPE; Task 8 covers final verification.
- Placeholder scan: The plan contains no unfinished markers or incomplete implementation slots. Runtime preference expansion beyond assignment is explicitly blocked until real labels and validated checkpoints exist.
- Type consistency: `ActionCandidate`, `DecisionSurface`, `candidate_features`, `candidate_feature_names`, `candidate_actions`, and `action_mask` are named consistently across environment, buffer, actor, and trainer tasks.
