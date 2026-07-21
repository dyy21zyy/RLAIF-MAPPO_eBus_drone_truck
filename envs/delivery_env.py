"""Deterministic event-driven delivery MDP for the Stage 3 simulator.

The environment intentionally has no Gymnasium dependency, but follows its reset
and step return signatures.  One call to :meth:`step` resolves exactly one
assignment or bus decision and then advances through automatic events.
"""

from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from envs.decision_schema import DecisionSurface
from envs.state_builder import (
    build_assignment_decision_surface,
    build_bus_charging_decision_surface,
    build_bus_loading_decision_surface,
    build_station_decision_surface,
    build_truck_decision_surface,
)
from utils.config import load_config
from envs.dynamics.bus_circulation import RuntimePhysicalBus, assert_no_overlaps

EPSILON = 1e-9
EVENT_PRIORITY = {
    "battery_ready": 0,
    "drone_return": 1,
    "station_operation": 2,
    "drone_dispatch": 3,
    "parcel_delivery": 4,
    "parcel_bus_terminal_arrival": 5,
    "parcel_station_arrival": 6,
    "bus_departure": 7,
    "bus_arrival": 8,
    "truck_departure": 6,
    "truck_arrive_stop": 6,
    "truck_unload": 6,
    "truck_route_complete": 6,
    "truck_available": 9,
    "parcel_release": 10,
}


class InstanceValidationError(ValueError):
    """Raised when a Stage 2 instance cannot safely initialize Stage 3."""


@dataclass(order=True)
class Event:
    """A stable event-queue entry ordered by time, priority, then insertion ID."""

    time_min: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class ParcelState:
    parcel_id: str
    release_time_min: float
    deadline_min: float
    weight_kg: float
    volume: float
    priority: int
    nearest_station_id: str
    drone_feasible: bool
    status: str = "UNRELEASED"
    action_id: int | None = None
    mode: str | None = None
    station_id: str | None = None
    truck_id: str | None = None
    delivered_time_min: float | None = None
    cost: float = 0.0


@dataclass
class StationState:
    station_id: str
    stop_id: str
    locker_capacity_kg: float
    drones: int
    full_batteries: int
    power_capacity_kw: float
    battery_power_kw: float
    battery_charge_duration_min: float
    locker_load_kg: float = 0.0
    active_bus_charges: list[float] = field(default_factory=list)
    active_battery_charges: list[tuple[float, float]] = field(default_factory=list)
    drone_available_min: list[float] = field(default_factory=list)
    battery_ready_min: list[float] = field(default_factory=list)


@dataclass
class TruckState:
    """Explicit state for batched multi-stop truck route execution."""

    truck_id: str
    current_location_id: str
    available_time: float
    remaining_capacity_kg: float
    remaining_volume_m3: float = 0.0
    onboard_parcels: list[str] = field(default_factory=list)
    total_distance: float = 0.0
    total_travel_time: float = 0.0
    status: str = "idle"
    route_history: list[list[str]] = field(default_factory=list)


@dataclass
class Decision:
    agent: str
    event: Event
    action_mask: list[bool]
    fallback_feasibility: bool = False


class DynamicDeliveryEnv:
    """Event-driven assignment and electric-bus charging environment.

    Assignment action IDs are stable: ``0`` is direct truck delivery,
    ``1..S`` are truck-bus-drone actions, and ``S+1..2S`` are
    truck-locker-drone actions. Bus action IDs index ``charging_actions_sec``.
    """

    metadata = {"name": "DynamicDeliveryEnv", "stage": 3, "schema_version": 1}

    def __init__(self, instance_path: str | Path, config_path: str | Path | None = None):
        self.instance_path = Path(instance_path)
        if self.instance_path.is_dir():
            self.instance_path = self.instance_path / "instance.json"
        if not self.instance_path.is_file():
            raise InstanceValidationError(f"Instance manifest does not exist: {self.instance_path}")
        self.instance_dir = self.instance_path.parent
        self.manifest = json.loads(self.instance_path.read_text(encoding="utf-8"))
        if self.manifest.get("stage") != 2:
            raise InstanceValidationError("Stage 3 requires a manifest with stage: 2")
        self.config = load_config(config_path) if config_path else self.manifest.get("config_snapshot")
        if not isinstance(self.config, dict):
            raise InstanceValidationError("Manifest has no valid config_snapshot; pass config_path")
        self._load_instance()
        self._initialised = False

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            raise InstanceValidationError(f"Required instance artifact is missing: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _as_bool(value: Any) -> bool:
        return str(value).lower() in {"true", "1", "yes"}

    def _artifact(self, key: str) -> Path:
        name = self.manifest.get("artifacts", {}).get(key)
        if not name:
            raise InstanceValidationError(f"Manifest is missing artifact mapping {key!r}")
        return self.instance_dir / name

    def _load_instance(self) -> None:
        parcel_rows = self._read_csv(self._artifact("parcels"))
        station_rows = self._read_csv(self._artifact("integrated_stations"))
        trip_rows = self._read_csv(self._artifact("bus_trips"))
        stop_time_rows = self._read_csv(self._artifact("bus_stop_times"))
        physical_rows = self._read_csv(self._artifact("physical_buses")) if self.manifest.get("artifacts", {}).get("physical_buses") else []
        trip_to_bus_rows = self._read_csv(self._artifact("trip_to_bus")) if self.manifest.get("artifacts", {}).get("trip_to_bus") else []
        if not parcel_rows or not station_rows or not trip_rows or not stop_time_rows:
            raise InstanceValidationError("Parcels, stations, trips, and stop times must be non-empty")

        self.parcel_rows = sorted(parcel_rows, key=lambda row: (float(row.get("release_time_min", row.get("release_time"))), row["parcel_id"]))
        self.station_rows = sorted(station_rows, key=lambda row: row["station_id"])
        self.station_ids = [row["station_id"] for row in self.station_rows]
        if len(self.station_ids) != len(set(self.station_ids)):
            raise InstanceValidationError("Station IDs must be unique")
        self.station_index = {station_id: index for index, station_id in enumerate(self.station_ids)}
        self.stop_to_station = {row["stop_id"]: row["station_id"] for row in self.station_rows}

        matrix_metadata = self.manifest.get("matrix_indices", {})
        truck_locations = matrix_metadata.get("truck_locations", [])
        self.truck_location_index = {row["id"]: index for index, row in enumerate(truck_locations)}
        self.drone_row_index = {row["station_id"]: int(row["index"]) for row in matrix_metadata.get("drone_rows", [])}
        self.drone_column_index = {row["parcel_id"]: int(row["index"]) for row in matrix_metadata.get("drone_columns", [])}
        required_locations = {"depot_01", *(row["parcel_id"] for row in parcel_rows), *self.station_ids}
        if not required_locations <= self.truck_location_index.keys():
            raise InstanceValidationError("Truck matrix metadata does not contain all required locations")
        if set(self.station_ids) != set(self.drone_row_index) or {row["parcel_id"] for row in parcel_rows} != set(self.drone_column_index):
            raise InstanceValidationError("Drone matrix metadata is inconsistent with station/parcel tables")

        self.truck_distance_m = np.load(self._artifact("truck_distance_matrix"), allow_pickle=False)
        self.truck_time_min = np.load(self._artifact("truck_travel_time_matrix"), allow_pickle=False)
        self.drone_distance_m = np.load(self._artifact("drone_distance_matrix"), allow_pickle=False)
        expected_truck = len(truck_locations)
        if self.truck_distance_m.shape != (expected_truck, expected_truck) or self.truck_time_min.shape != (expected_truck, expected_truck):
            raise InstanceValidationError("Truck matrix shape does not match manifest indices")
        if self.drone_distance_m.shape != (len(self.station_ids), len(parcel_rows)):
            raise InstanceValidationError("Drone matrix shape does not match manifest indices")
        if not all(np.isfinite(matrix).all() and (matrix >= 0).all() for matrix in (self.truck_distance_m, self.truck_time_min, self.drone_distance_m)):
            raise InstanceValidationError("Distance and time matrices must contain finite non-negative values")

        self.trip_rows = {row["trip_id"]: row for row in trip_rows}
        self.physical_bus_rows = physical_rows
        self.trip_to_bus_rows = trip_to_bus_rows
        self.trip_to_bus = {row["trip_id"]: row["bus_id"] for row in trip_to_bus_rows}
        if trip_to_bus_rows:
            if set(self.trip_to_bus) != set(self.trip_rows):
                raise InstanceValidationError("Every scheduled trip must map to exactly one physical bus")
            try:
                assert_no_overlaps(trip_to_bus_rows)
            except ValueError as exc:
                raise InstanceValidationError(str(exc)) from exc
        self.data_sources = self._load_data_sources()
        self.trip_stop_times: dict[str, list[dict[str, str]]] = {}
        for row in stop_time_rows:
            self.trip_stop_times.setdefault(row["trip_id"], []).append(row)
        for rows in self.trip_stop_times.values():
            rows.sort(key=lambda row: int(row["stop_sequence"]))
        missing_trips = set(self.trip_rows) - set(self.trip_stop_times)
        if missing_trips:
            raise InstanceValidationError(f"Trips have no stop times: {sorted(missing_trips)}")

    def _load_data_sources(self) -> dict[str, dict[str, Any]]:
        """Return source-provenance entries keyed for observations/RLAIF context."""

        entries: dict[str, dict[str, Any]] = {}
        manifest_sources = self.manifest.get("data_provenance", {})
        if isinstance(manifest_sources, dict):
            for field, value in manifest_sources.items():
                key = str(field).replace(" ", "_")
                if isinstance(value, dict):
                    entries[key] = value
        provenance_name = self.manifest.get("artifacts", {}).get("data_provenance")
        if provenance_name:
            provenance_path = self.instance_dir / provenance_name
            if provenance_path.is_file():
                try:
                    loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    loaded = {}
                for entry in loaded.get("entries", []):
                    field = str(entry.get("field", "")).replace(" ", "_")
                    if field:
                        entries[field] = {
                            "source_type": entry.get("source_type", ""),
                            "source_file": entry.get("source_file", ""),
                            "notes": entry.get("notes", ""),
                            "value": entry.get("value"),
                        }
        return entries

    @property
    def assignment_action_size(self) -> int:
        return 1 + 2 * len(self.station_ids)

    @property
    def bus_action_size(self) -> int:
        return len(self.config["bus"]["charging_actions_sec"])

    @property
    def truck_available_min(self) -> list[float]:
        """Compatibility view backed by explicit truck states."""
        return [truck.available_time for truck in self.trucks]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all mutable state and advance to the first decision epoch."""
        del options  # Reserved for Gymnasium compatibility.
        self.seed = int(self.config["project"]["seed"] if seed is None else seed)
        self.now_min = 0.0
        self.horizon_min = float(self.config["bus"]["delivery_horizon_min"])
        self.event_sequence = 0
        self.events: list[Event] = []
        self.current_decision: Decision | None = None
        self.terminated = False
        self.truncated = False
        truck_capacity = float(self.config["truck"].get("weight_capacity_kg", self.config["truck"].get("capacity_kg", 100.0)))
        self.trucks = [
            TruckState(
                truck_id=f"truck_{index:03d}",
                current_location_id="depot_01",
                available_time=0.0,
                remaining_capacity_kg=truck_capacity,
                remaining_volume_m3=float(self.config["truck"].get("volume_capacity_m3", 1.0)),
            )
            for index in range(int(self.config["truck"]["num_trucks"]))
        ]
        self.physical_buses: dict[str, RuntimePhysicalBus] = {}
        for row in (self.physical_bus_rows or []):
            self.physical_buses[row["bus_id"]] = RuntimePhysicalBus(row["bus_id"], row.get("initial_location_id", "terminal"), float(row.get("initial_soc_kwh", self.config["bus"]["bus_battery_kwh"])), float(row.get("battery_capacity_kwh", self.config["bus"]["bus_battery_kwh"])), float(row.get("minimum_safe_energy_kwh", self.config["bus"]["bus_min_soc_kwh"])))
        if not self.physical_buses:
            self.physical_buses = {f"bus_{i:03d}": RuntimePhysicalBus(f"bus_{i:03d}", "terminal", float(self.config["bus"]["bus_battery_kwh"])) for i, _ in enumerate(self.trip_rows)}
            self.trip_to_bus = {tid: f"bus_{i:03d}" for i, tid in enumerate(self.trip_rows)}
        self.bus_soc_kwh = {trip_id: self.physical_buses[self.trip_to_bus[trip_id]].soc_kwh for trip_id in self.trip_rows}
        self.bus_delay_min = {trip_id: self.physical_buses[self.trip_to_bus[trip_id]].schedule_delay_min for trip_id in self.trip_rows}
        self.bus_freight_kg: dict[str, float] = {trip_id: 0.0 for trip_id in self.trip_rows}
        self.bus_segment_index: dict[str, int] = {trip_id: 0 for trip_id in self.trip_rows}
        self.pending_bus_parcels: dict[tuple[str, str], list[str]] = {}
        self.pending_truck_tasks: list[dict[str, Any]] = []  # legacy mirror; formal truck decisions use WAITING_TRUCK parcel pool
        self.pending_truck_decision_min: dict[int, float] = {}
        self.bus_terminal_ready: dict[str, list[str]] = {}
        self.waiting_station_parcels: dict[str, list[str]] = {}
        self.reward_total = 0.0
        self.decision_counts = {"assignment": 0, "truck": 0, "bus": 0, "station": 0}
        self.infeasible_action_corrections = 0
        self.fallback_feasibility_events = 0
        self.accumulated_power_overload = 0.0
        self.accumulated_power_overload_duration = 0.0
        self.accumulated_locker_overflow = 0.0
        self.accumulated_locker_overflow_duration = 0.0
        self.peak_station_load_kw = max((float(row["base_load_kw"]) if "base_load_kw" in row else float(self.config["station"]["base_load_kw"]) for row in self.station_rows), default=0.0)
        self.truck_dispatch_count = 0
        self.truck_parcels_routed = 0
        self.truck_weight_utilization_sum = 0.0
        self.truck_volume_utilization_sum = 0.0
        self.cost_components = {name: 0.0 for name in (
            "passenger_delay", "bus_operating_delay", "parcel_lateness", "energy_cost", "power_overload",
            "bus_battery_violation", "locker_overflow", "truck_cost", "undelivered", "battery_shortage", "infeasible_action")}
        self.parcels = {
            row["parcel_id"]: ParcelState(
                parcel_id=row["parcel_id"], release_time_min=float(row.get("release_time_min", row.get("release_time"))),
                deadline_min=float(row.get("deadline_min", row.get("deadline"))), weight_kg=float(row.get("weight_kg", row.get("weight"))),
                volume=float(row.get("volume_m3", row.get("volume"))),
                priority=int(row["priority"]), nearest_station_id=row["nearest_station_id"],
                drone_feasible=self._as_bool(row["drone_feasible"]),
            ) for row in self.parcel_rows
        }
        self.stations = {}
        for row in self.station_rows:
            drones = int(row["drones_num"])
            self.stations[row["station_id"]] = StationState(
                station_id=row["station_id"], stop_id=row["stop_id"], locker_capacity_kg=float(row["locker_capacity_kg"]),
                drones=drones, full_batteries=int(row["initial_full_batteries"]), power_capacity_kw=float(row["power_capacity_kw"]),
                battery_power_kw=float(row["battery_charging_power_kw"]), battery_charge_duration_min=float(row["battery_charging_duration_min"]),
                drone_available_min=[0.0] * drones,
            )
        for parcel in self.parcels.values():
            self._push(parcel.release_time_min, "parcel_release", {"parcel_id": parcel.parcel_id})
        for trip_id, rows in self.trip_stop_times.items():
            if self._as_bool(self.trip_rows[trip_id]["freight_allowed"]):
                self._push(float(rows[0]["departure_time"]), "bus_departure", {"trip_id": trip_id})
            first_station_index = next((i for i, row in enumerate(rows) if row["stop_id"] in self.stop_to_station), None)
            if first_station_index is not None:
                self.bus_segment_index[trip_id] = first_station_index
                self._push(float(rows[first_station_index]["arrival_time"]), "bus_arrival", {"trip_id": trip_id, "stop_index": first_station_index})
        self._initialised = True
        reward = self._advance()
        self.reward_total += reward
        return self._observation(), self._info(reward)

    def _push(self, time_min: float, kind: str, payload: dict[str, Any]) -> None:
        heapq.heappush(self.events, Event(float(time_min), EVENT_PRIORITY[kind], self.event_sequence, kind, payload))
        self.event_sequence += 1

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one discrete action and advance to the next decision."""
        if not self._initialised:
            raise RuntimeError("Call reset() before step()")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is complete; call reset()")
        if self.current_decision is None:
            raise RuntimeError("No decision is pending")
        decision = self.current_decision
        self.current_decision = None
        self.decision_counts[decision.agent] += 1
        reward = 0.0
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= len(decision.action_mask):
            raise ValueError(f"Action must be an integer in [0, {len(decision.action_mask) - 1}]")
        selected = int(action)
        corrected = False
        if not decision.action_mask[selected]:
            selected = next(index for index, feasible in enumerate(decision.action_mask) if feasible)
            reward += self._charge_cost("infeasible_action", 1.0)
            corrected = True
            self.infeasible_action_corrections += 1
        if decision.fallback_feasibility:
            if not corrected:
                reward += self._charge_cost("infeasible_action", 1.0)
                corrected = True
                self.infeasible_action_corrections += 1
            parcel = self.parcels[decision.event.payload["parcel_id"]]
            parcel.action_id, parcel.mode, parcel.status = 0, "TD", "FAILED"
        elif decision.agent == "assignment":
            reward += self._apply_assignment_decision(decision.event.payload["parcel_id"], selected)
        elif decision.agent == "truck":
            reward += self._apply_truck_action(decision.event, selected)
        elif decision.agent == "bus" and decision.event.kind == "bus_departure":
            reward += self._apply_bus_loading_action(decision.event, selected)
        elif decision.agent == "bus":
            reward += self._apply_bus_action(decision.event, selected)
        elif decision.agent == "station":
            reward += self._apply_station_action(decision.event, selected)
        else:
            raise ValueError(f"Unknown decision agent: {decision.agent}")
        reward += self._advance()
        self.reward_total += reward
        info = self._info(reward)
        info.update({"requested_action": int(action), "applied_action": selected, "action_corrected": corrected})
        return self._observation(), reward, self.terminated, self.truncated, info

    def _advance(self) -> float:
        reward = 0.0
        while self.events and self.current_decision is None:
            event = heapq.heappop(self.events)
            target_time = min(max(self.now_min, event.time_min), self.horizon_min)
            reward += self._integrate_station_penalties(self.now_min, target_time)
            self._refresh_truck_states(target_time)
            if event.time_min > self.horizon_min + EPSILON:
                self.now_min = self.horizon_min
                break
            self.now_min = target_time
            if event.kind == "parcel_release":
                parcel = self.parcels[event.payload["parcel_id"]]
                parcel.status = "PENDING_ASSIGNMENT"
                mask, fallback = self._assignment_feasibility(parcel)
                if fallback:
                    self.fallback_feasibility_events += 1
                self.current_decision = Decision("assignment", event, mask, fallback)
            elif event.kind == "truck_available":
                truck_index = int(event.payload["truck_index"])
                scheduled_time = self.pending_truck_decision_min.get(truck_index)
                if scheduled_time is None or abs(scheduled_time - event.time_min) > EPSILON:
                    continue
                self.pending_truck_decision_min.pop(truck_index, None)
                if not any(p.status == "WAITING_TRUCK" for p in self.parcels.values()):
                    continue
                truck = self.trucks[truck_index]
                if truck.available_time > self.now_min + EPSILON:
                    self._push_next_truck_decision()
                    continue
                surface = build_truck_decision_surface(self, truck)
                self.current_decision = Decision("truck", event, surface.action_mask())
            elif event.kind == "truck_departure":
                pass
            elif event.kind == "truck_arrive_stop":
                self.trucks[[t.truck_id for t in self.trucks].index(event.payload["truck_id"])].current_location_id = event.payload["stop_id"]
            elif event.kind == "truck_unload":
                st=event.payload["stop_type"]
                for pid in event.payload.get("parcel_ids", []):
                    parcel=self.parcels[pid]
                    if st == "CUSTOMER":
                        parcel.status, parcel.delivered_time_min = "DELIVERED", self.now_min
                        lateness=max(0.0,self.now_min-parcel.deadline_min)*parcel.priority
                        parcel.cost += lateness; reward += self._charge_cost("parcel_lateness", lateness)
                    elif st == "BUS_TERMINAL":
                        parcel.status = "AT_BUS_TERMINAL"
                        self.bus_terminal_ready.setdefault(parcel.station_id or "", []).append(pid)
                    elif st == "INTEGRATED_STATION":
                        reward += self._handle_station_arrival(pid, parcel.station_id or event.payload["stop_id"])
            elif event.kind == "truck_route_complete":
                truck=self.trucks[[t.truck_id for t in self.trucks].index(event.payload["truck_id"])]
                truck.current_location_id=event.payload["final_location_id"]; truck.onboard_parcels.clear(); truck.status="idle"
                truck.remaining_capacity_kg=float(self.config["truck"].get("weight_capacity_kg", self.config["truck"].get("capacity_kg",100.0)))
                truck.remaining_volume_m3=float(self.config["truck"].get("volume_capacity_m3",1.0))
            elif event.kind == "bus_departure":
                trip_id = event.payload["trip_id"]
                bus = self.physical_buses[self.trip_to_bus[trip_id]]
                if bus.next_available_time_min > self.now_min + EPSILON:
                    self._push(bus.next_available_time_min, "bus_departure", event.payload)
                    continue
                scheduled = float(self.trip_stop_times[trip_id][0]["departure_time"])
                bus.current_trip_id = trip_id
                bus.schedule_delay_min += max(0.0, self.now_min - scheduled)
                self.bus_delay_min[trip_id] = bus.schedule_delay_min
                surface = build_bus_loading_decision_surface(self, trip_id)
                self.current_decision = Decision("bus", event, surface.action_mask())
            elif event.kind == "bus_arrival":
                reward += self._handle_bus_arrival(event)
                surface = build_bus_charging_decision_surface(self, event)
                self.current_decision = Decision("bus", event, surface.action_mask())
            elif event.kind == "parcel_delivery":
                parcel = self.parcels[event.payload["parcel_id"]]
                parcel.status, parcel.delivered_time_min = "DELIVERED", self.now_min
                lateness = max(0.0, self.now_min - parcel.deadline_min) * parcel.priority
                parcel.cost += lateness
                reward += self._charge_cost("parcel_lateness", lateness)
            elif event.kind == "parcel_bus_terminal_arrival":
                parcel = self.parcels[event.payload["parcel_id"]]
                parcel.status = "AT_BUS_TERMINAL"
                self.bus_terminal_ready.setdefault(parcel.station_id or event.payload.get("station_id", ""), []).append(parcel.parcel_id)
            elif event.kind == "parcel_station_arrival":
                reward += self._handle_station_arrival(event.payload["parcel_id"], event.payload["station_id"])
            elif event.kind == "station_operation":
                surface = build_station_decision_surface(self, event.payload["station_id"])
                self.current_decision = Decision("station", event, surface.action_mask())
            elif event.kind == "drone_dispatch":
                reward += self._dispatch_drone(
                    event.payload["parcel_id"],
                    event.payload["station_id"],
                    event.payload["drone_index"],
                    bool(event.payload["battery_reserved"]),
                )
            elif event.kind == "drone_return":
                station = self.stations[event.payload["station_id"]]
                station.drone_available_min[event.payload["drone_index"]] = self.now_min
                if self.waiting_station_parcels.get(station.station_id):
                    self._push(self.now_min, "station_operation", {"station_id": station.station_id})
            elif event.kind == "battery_ready":
                station = self.stations[event.payload["station_id"]]
                station.full_batteries += 1
                if station.battery_ready_min:
                    heapq.heappop(station.battery_ready_min)
                station.active_battery_charges = [
                    session for session in station.active_battery_charges
                    if session[1] > self.now_min + EPSILON
                ]
                if self.waiting_station_parcels.get(station.station_id):
                    self._push(self.now_min, "station_operation", {"station_id": station.station_id})
        if self.current_decision is None and (not self.events or self.now_min >= self.horizon_min - EPSILON):
            if self.now_min < self.horizon_min:
                reward += self._integrate_station_penalties(self.now_min, self.horizon_min)
                self.now_min = self.horizon_min
            reward += self._finish_episode()
        return reward

    def _refresh_truck_states(self, time_min: float) -> None:
        capacity = float(self.config["truck"]["capacity_kg"])
        for truck in self.trucks:
            if truck.available_time <= time_min + EPSILON:
                truck.current_location_id = truck.route_history[-1][-1] if truck.route_history else "depot_01"
                truck.remaining_capacity_kg = capacity
                truck.remaining_volume_m3 = float(self.config["truck"].get("volume_capacity_m3", 1.0))
                truck.onboard_parcels.clear()
                truck.status = "idle"

    def _earliest_truck(self) -> TruckState:
        return min(self.trucks, key=lambda truck: (truck.available_time, truck.truck_id))

    def _earliest_truck_index(self) -> int:
        return min(
            range(len(self.trucks)),
            key=lambda index: (self.trucks[index].available_time, self.trucks[index].truck_id),
        )

    def _push_next_truck_decision(self) -> None:
        if self.trucks and self.pending_truck_tasks:
            index = self._earliest_truck_index()
            decision_time = max(self.now_min, self.trucks[index].available_time)
            scheduled_time = self.pending_truck_decision_min.get(index)
            if scheduled_time is not None and scheduled_time <= decision_time + EPSILON:
                return
            self.pending_truck_decision_min[index] = decision_time
            self._push(
                decision_time,
                "truck_available",
                {"truck_index": index},
            )

    def _record_truck_trip(
        self,
        truck: TruckState,
        parcel: ParcelState,
        route: list[str],
        available_time: float,
        distance_km: float,
        travel_time_min: float,
    ) -> None:
        capacity = float(self.config["truck"]["capacity_kg"])
        truck.available_time = available_time
        truck.remaining_capacity_kg = max(0.0, capacity - parcel.weight_kg)
        truck.onboard_parcels = [parcel.parcel_id]
        truck.total_distance += distance_km
        truck.total_travel_time += travel_time_min
        truck.status = "traveling"
        truck.current_location_id = route[0]
        truck.route_history.append(route)
        parcel.truck_id = truck.truck_id

    def _apply_assignment_decision(self, parcel_id: str, action: int) -> float:
        """Convert assignment choices into truck tasks for four-agent control."""
        parcel = self.parcels[parcel_id]
        parcel.action_id = action
        if action == 0:
            parcel.mode = "TD"
            depot = self.truck_location_index["depot_01"]
            customer = self.truck_location_index[parcel_id]
            travel = float(self.truck_time_min[depot, customer])
            return_time = (
                float(self.truck_time_min[customer, depot])
                if self.config["truck"]["return_to_depot"]
                else 0.0
            )
            service = float(self.config["network"]["customer_service_time_min"])
            parcel.status = "WAITING_TRUCK"
            self.pending_truck_tasks.append(
                {
                    "kind": "direct_delivery",
                    "parcel_id": parcel_id,
                    "estimated_time_min": travel + service + return_time,
                }
            )
            self._push_next_truck_decision()
            return 0.0

        station_offset = action - 1
        if station_offset < len(self.station_ids):
            parcel.mode = "TBD"
            station_id = self.station_ids[station_offset]
            trip = self._next_freight_trip(
                self.now_min,
                station_id,
                parcel.weight_kg,
                min(parcel.deadline_min, self.horizon_min),
            )
            if trip is None:
                parcel.status = "FAILED"
                return self._charge_cost("infeasible_action", 1.0)
            trip_id, _arrival = trip
            rows = self.trip_stop_times[trip_id]
            terminal_stop = rows[0]["stop_id"]
            terminal_location = self.truck_location_index[terminal_stop]
            depot = self.truck_location_index["depot_01"]
            travel = float(self.truck_time_min[depot, terminal_location])
            return_time = (
                float(self.truck_time_min[terminal_location, depot])
                if self.config["truck"]["return_to_depot"]
                else 0.0
            )
            parcel.status, parcel.station_id = "WAITING_TRUCK", station_id
            self.pending_truck_tasks.append(
                {
                    "kind": "bus_terminal_feeder",
                    "parcel_id": parcel_id,
                    "terminal_transfer_required": True,
                    "station_id": station_id,
                    "terminal_stop_id": terminal_stop,
                    "estimated_time_min": travel + return_time,
                }
            )
            self._push_next_truck_decision()
            return 0.0

        parcel.mode = "TLD"
        station_id = self.station_ids[station_offset - len(self.station_ids)]
        parcel.status, parcel.station_id = "WAITING_TRUCK", station_id
        depot = self.truck_location_index["depot_01"]
        station = self.truck_location_index[station_id]
        travel = float(self.truck_time_min[depot, station])
        return_time = (
            float(self.truck_time_min[station, depot])
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        self.pending_truck_tasks.append(
            {
                "kind": "station_feeder",
                "parcel_id": parcel_id,
                "station_id": station_id,
                "estimated_time_min": travel + return_time,
            }
        )
        self._push_next_truck_decision()
        return 0.0

    def _apply_truck_action(self, event: Event, action: int) -> float:
        truck = self.trucks[int(event.payload["truck_index"])]
        from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates
        from envs.dynamics.truck_dynamics import apply_truck_batch
        surface_candidates = generate_truck_batch_candidates(self, truck)
        if action >= len(surface_candidates):
            return 0.0
        candidate = surface_candidates[action]
        if not candidate.feasible:
            return self._charge_cost("infeasible_action", 1.0)
        if not candidate.idle_flag:
            for pid in candidate.parcel_ids:
                self.pending_truck_tasks = [t for t in self.pending_truck_tasks if t.get("parcel_id") != pid]
        return apply_truck_batch(self, truck, candidate)
        task = self.pending_truck_tasks.pop(action)
        parcel = self.parcels[task["parcel_id"]]
        if parcel.weight_kg > truck.remaining_capacity_kg + EPSILON:
            self.pending_truck_tasks.insert(action, task)
            return self._charge_cost("infeasible_action", 1.0)
        if task["kind"] == "direct_delivery":
            return self._execute_direct_truck_task(truck, parcel)
        if task["kind"] == "station_feeder":
            return self._execute_station_feeder_task(truck, parcel, task["station_id"])
        if task["kind"] == "bus_terminal_feeder":
            return self._execute_bus_terminal_feeder_task(truck, parcel, task)
        return self._charge_cost("infeasible_action", 1.0)

    def _execute_direct_truck_task(self, truck: TruckState, parcel: ParcelState) -> float:
        start = max(self.now_min, truck.available_time) + parcel.weight_kg * float(
            self.config["truck"]["loading_time_min_per_kg"]
        )
        depot = self.truck_location_index["depot_01"]
        customer = self.truck_location_index[parcel.parcel_id]
        travel = float(self.truck_time_min[depot, customer])
        outbound_distance_km = float(self.truck_distance_m[depot, customer]) / 1000
        return_distance_km = (
            float(self.truck_distance_m[customer, depot]) / 1000
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        service = float(self.config["network"]["customer_service_time_min"])
        return_time = (
            float(self.truck_time_min[customer, depot])
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        completion = start + travel + service
        available_time = completion + return_time
        parcel.status = "ONBOARD_TRUCK"
        self._push(completion, "parcel_delivery", {"parcel_id": parcel.parcel_id})
        route = ["depot_01", parcel.parcel_id]
        if self.config["truck"]["return_to_depot"]:
            route.append("depot_01")
        distance_km = outbound_distance_km + return_distance_km
        self._record_truck_trip(
            truck, parcel, route, available_time, distance_km, travel + return_time
        )
        truck_cost = (
            float(self.config["truck"]["fixed_dispatch_cost"])
            + distance_km * float(self.config["truck"]["cost_per_km"])
            + (travel + return_time) * float(self.config["truck"]["cost_per_min"])
        )
        self._push_next_truck_decision()
        return self._charge_cost("truck_cost", truck_cost)

    def _execute_station_feeder_task(
        self, truck: TruckState, parcel: ParcelState, station_id: str
    ) -> float:
        start = max(self.now_min, truck.available_time) + parcel.weight_kg * float(
            self.config["truck"]["loading_time_min_per_kg"]
        )
        depot = self.truck_location_index["depot_01"]
        station = self.truck_location_index[station_id]
        travel = float(self.truck_time_min[depot, station])
        return_time = (
            float(self.truck_time_min[station, depot])
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        arrival = start + travel + parcel.weight_kg * float(
            self.config["truck"]["unloading_time_min_per_kg"]
        )
        available_time = arrival + return_time
        parcel.status = "ONBOARD_TRUCK"
        self._push(arrival, "parcel_station_arrival", {"parcel_id": parcel.parcel_id, "station_id": station_id})
        outbound_distance_km = float(self.truck_distance_m[depot, station]) / 1000
        return_distance_km = (
            float(self.truck_distance_m[station, depot]) / 1000
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        distance_km = outbound_distance_km + return_distance_km
        route = ["depot_01", station_id]
        if self.config["truck"]["return_to_depot"]:
            route.append("depot_01")
        self._record_truck_trip(
            truck, parcel, route, available_time, distance_km, travel + return_time
        )
        truck_cost = (
            float(self.config["truck"]["fixed_dispatch_cost"])
            + distance_km * float(self.config["truck"]["cost_per_km"])
            + (travel + return_time) * float(self.config["truck"]["cost_per_min"])
        )
        self._push_next_truck_decision()
        return self._charge_cost("truck_cost", truck_cost)

    def _execute_bus_terminal_feeder_task(
        self, truck: TruckState, parcel: ParcelState, task: dict[str, Any]
    ) -> float:
        terminal_stop = task["terminal_stop_id"]
        start = max(self.now_min, truck.available_time) + parcel.weight_kg * float(
            self.config["truck"]["loading_time_min_per_kg"]
        )
        depot = self.truck_location_index["depot_01"]
        terminal = self.truck_location_index[terminal_stop]
        travel = float(self.truck_time_min[depot, terminal])
        return_time = (
            float(self.truck_time_min[terminal, depot])
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        arrival = start + travel + parcel.weight_kg * float(
            self.config["truck"]["unloading_time_min_per_kg"]
        )
        available_time = arrival + return_time
        parcel.status = "ONBOARD_TRUCK"
        self._push(
            arrival,
            "parcel_bus_terminal_arrival",
            {"parcel_id": parcel.parcel_id, "station_id": task["station_id"]},
        )
        outbound_distance_km = float(self.truck_distance_m[depot, terminal]) / 1000
        return_distance_km = (
            float(self.truck_distance_m[terminal, depot]) / 1000
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        distance_km = outbound_distance_km + return_distance_km
        route = ["depot_01", terminal_stop]
        if self.config["truck"]["return_to_depot"]:
            route.append("depot_01")
        self._record_truck_trip(
            truck, parcel, route, available_time, distance_km, travel + return_time
        )
        truck_cost = (
            float(self.config["truck"]["fixed_dispatch_cost"])
            + distance_km * float(self.config["truck"]["cost_per_km"])
            + (travel + return_time) * float(self.config["truck"]["cost_per_min"])
        )
        self._push_next_truck_decision()
        return self._charge_cost("truck_cost", truck_cost)

    def _station_load_kw(self, station: StationState, time_min: float) -> float:
        bus_charges = sum(end > time_min + EPSILON for end in station.active_bus_charges)
        battery_charges = sum(
            start <= time_min + EPSILON and end > time_min + EPSILON
            for start, end in station.active_battery_charges
        )
        return (
            float(self.config["station"]["base_load_kw"])
            + bus_charges * float(self.config["bus"]["charging_power_kw"])
            + battery_charges * station.battery_power_kw
        )

    def _integrate_station_penalties(self, previous_time: float, current_time: float) -> float:
        """Integrate piecewise-constant station overload and overflow in minutes."""
        start, end = float(previous_time), float(current_time)
        if end <= start + EPSILON:
            return 0.0
        power_amount = power_duration = locker_amount = locker_duration = 0.0
        for station in self.stations.values():
            boundaries = {start, end}
            boundaries.update(
                charge_end for charge_end in station.active_bus_charges
                if start < charge_end < end
            )
            for charge_start, charge_end in station.active_battery_charges:
                if start < charge_start < end:
                    boundaries.add(charge_start)
                if start < charge_end < end:
                    boundaries.add(charge_end)
            ordered = sorted(boundaries)
            for segment_start, segment_end in zip(ordered, ordered[1:]):
                duration = segment_end - segment_start
                station_load_kw = self._station_load_kw(station, segment_start)
                self.peak_station_load_kw = max(self.peak_station_load_kw, station_load_kw)
                overload_kw = max(0.0, station_load_kw - station.power_capacity_kw)
                overflow_kg = max(0.0, station.locker_load_kg - station.locker_capacity_kg)
                power_amount += overload_kw * duration
                locker_amount += overflow_kg * duration
                if overload_kw > EPSILON:
                    power_duration += duration
                if overflow_kg > EPSILON:
                    locker_duration += duration
        self.accumulated_power_overload += power_amount
        self.accumulated_power_overload_duration += power_duration
        self.accumulated_locker_overflow += locker_amount
        self.accumulated_locker_overflow_duration += locker_duration
        return self._charge_cost("power_overload", power_amount) + self._charge_cost(
            "locker_overflow", locker_amount
        )

    def _assignment_mask(self, parcel: ParcelState) -> list[bool]:
        """Return hard-feasibility mask with TD exposed as the last-resort fallback."""
        return self._assignment_feasibility(parcel)[0]

    def _assignment_feasibility(self, parcel: ParcelState) -> tuple[list[bool], bool]:
        truck_capacity = float(self.config["truck"].get("weight_capacity_kg", self.config["truck"].get("capacity_kg", 100.0)))
        has_capable_truck = bool(self.truck_available_min) and parcel.weight_kg <= truck_capacity + EPSILON
        depot = self.truck_location_index["depot_01"]
        customer = self.truck_location_index[parcel.parcel_id]
        loading = parcel.weight_kg * float(self.config["truck"]["loading_time_min_per_kg"])
        earliest_start = max(self.now_min, min(self.truck_available_min)) + loading if has_capable_truck else float("inf")
        direct_completion = earliest_start + float(self.truck_time_min[depot, customer]) + float(
            self.config["network"]["customer_service_time_min"]
        )
        direct_return = (
            float(self.truck_time_min[customer, depot])
            if self.config["truck"]["return_to_depot"]
            else 0.0
        )
        mask = [has_capable_truck and direct_completion + direct_return <= self.horizon_min + EPSILON]
        for station_id in self.station_ids:
            drone_feasible = self._station_can_serve_by_drone(parcel, station_id)
            latest_arrival = min(parcel.deadline_min, self.horizon_min)
            mask.append(
                has_capable_truck
                and drone_feasible
                and self._next_freight_trip(
                    self.now_min, station_id, parcel.weight_kg, latest_arrival
                ) is not None
            )
        for station_id in self.station_ids:
            station_state = self.stations.get(station_id)
            station_location = self.truck_location_index.get(station_id)
            truck_arrival = (
                earliest_start
                + float(self.truck_time_min[depot, station_location])
                + parcel.weight_kg * float(self.config["truck"]["unloading_time_min_per_kg"])
                if has_capable_truck and station_location is not None
                else float("inf")
            )
            locker_remaining = (
                station_state.locker_capacity_kg - station_state.locker_load_kg
                if station_state is not None
                else -1.0
            )
            mask.append(
                station_state is not None
                and self._station_can_serve_by_drone(parcel, station_id)
                and locker_remaining + EPSILON >= parcel.weight_kg
                and truck_arrival <= self.horizon_min + EPSILON
            )
        fallback = not any(mask)
        if fallback:
            mask[0] = True  # The correction path can still record an eventual undelivered parcel.
        return mask, fallback

    def _station_can_serve_by_drone(self, parcel: ParcelState, station_id: str) -> bool:
        if station_id not in self.stations or station_id not in self.drone_row_index:
            return False
        if not parcel.drone_feasible or parcel.weight_kg > float(self.config["network"]["drone_payload_kg"]) + EPSILON:
            return False
        distance_km = float(
            self.drone_distance_m[
                self.drone_row_index[station_id], self.drone_column_index[parcel.parcel_id]
            ]
        ) / 1000.0
        round_trip_min = (
            2.0 * distance_km / max(float(self.config["network"]["drone_speed_kmph"]), EPSILON) * 60.0
            + float(self.config["network"]["customer_service_time_min"])
        )
        return (
            distance_km <= float(self.config["network"]["drone_radius_km"]) + EPSILON
            and round_trip_min <= float(self.config["network"]["max_drone_round_trip_min"]) + EPSILON
        )

    def _next_freight_trip(
        self,
        ready_min: float,
        station_id: str,
        weight_kg: float,
        latest_arrival_min: float | None = None,
    ) -> tuple[str, float] | None:
        station = self.stations.get(station_id)
        if station is None or not self.truck_available_min or weight_kg > float(self.config["truck"]["capacity_kg"]) + EPSILON:
            return None
        station_stop = station.stop_id
        loading = weight_kg * float(self.config["bus"]["terminal_loading_time_min_per_kg"])
        candidates = []
        for trip_id, trip in self.trip_rows.items():
            if not self._as_bool(trip["freight_allowed"]):
                continue
            rows = self.trip_stop_times[trip_id]
            terminal_time = float(rows[0]["departure_time"])
            terminal_location = self.truck_location_index.get(rows[0]["stop_id"])
            if terminal_location is None:
                continue
            truck_loading = weight_kg * float(self.config["truck"]["loading_time_min_per_kg"])
            depot = self.truck_location_index["depot_01"]
            terminal_ready = (
                max(ready_min, min(self.truck_available_min))
                + truck_loading
                + float(self.truck_time_min[depot, terminal_location])
            )
            target = next((row for row in rows if row["stop_id"] == station_stop), None)
            target_arrival = float(target["arrival_time"]) if target else float("inf")
            if (
                target
                and terminal_ready + loading <= terminal_time + EPSILON
                and target_arrival <= self.horizon_min + EPSILON
                and (latest_arrival_min is None or target_arrival <= latest_arrival_min + EPSILON)
                and self.bus_freight_kg[trip_id] + weight_kg
                <= float(self.config["bus"]["freight_capacity_kg"]) + EPSILON
            ):
                candidates.append((target_arrival, trip_id))
        candidates.sort()
        return (candidates[0][1], candidates[0][0]) if candidates else None

    def _apply_assignment(self, parcel_id: str, action: int) -> float:
        """Compatibility helper used by legacy unit tests.

        Runtime assignment uses `_apply_assignment_decision` and creates truck tasks.
        This helper preserves direct execution for TD/TLD tests while keeping TBD
        free of assignment-time trip or vehicle binding.
        """
        parcel = self.parcels[parcel_id]
        parcel.action_id = action
        if action == 0:
            parcel.mode = "TD"
            return self._execute_direct_truck_task(self._earliest_truck(), parcel)
        station_offset = action - 1
        if station_offset < len(self.station_ids):
            parcel.mode = "TBD"
            parcel.station_id = self.station_ids[station_offset]
            parcel.status = "WAITING_TRUCK"
            self.pending_truck_tasks.append({
                "kind": "bus_terminal_feeder",
                "parcel_id": parcel_id,
                "station_id": parcel.station_id,
                "terminal_transfer_required": True,
                "terminal_stop_id": next(iter(self.truck_location_index)),
                "estimated_time_min": 0.0,
            })
            return 0.0
        parcel.mode = "TLD"
        parcel.station_id = self.station_ids[station_offset - len(self.station_ids)]
        return self._execute_station_feeder_task(self._earliest_truck(), parcel, parcel.station_id)

    def _handle_bus_arrival(self, event: Event) -> float:
        trip_id, stop_index = event.payload["trip_id"], event.payload["stop_index"]
        rows = self.trip_stop_times[trip_id]
        station_id = self.stop_to_station[rows[stop_index]["stop_id"]]
        event.payload["station_id"] = station_id
        reward = 0.0
        previous_stop_index = event.payload.get("previous_stop_index")
        if previous_stop_index is not None:
            previous, current = rows[int(previous_stop_index)], rows[stop_index]
            scheduled_segment_min = max(float(current["arrival_time"]) - float(previous["departure_time"]), 0.0)
            segment_km = scheduled_segment_min / 60 * float(self.config["bus"]["bus_speed_kmph"])
            bus = self.physical_buses[self.trip_to_bus[trip_id]]
            bus.current_trip_id = trip_id
            bus.soc_kwh -= segment_km * float(self.config["bus"]["bus_energy_kwh_per_km"])
            bus.current_location = current["stop_id"]
            bus.minimum_safe_energy_violation = bus.minimum_safe_energy_violation or bus.soc_kwh < bus.minimum_safe_energy_kwh
            if bus.soc_kwh <= 0:
                bus.depleted = True
                raise RuntimeError(f"Complete depletion for {bus.physical_bus_id} on {trip_id}")
            self.bus_soc_kwh[trip_id] = bus.soc_kwh
            shortage = max(0.0, float(self.config["bus"]["bus_min_soc_kwh"]) - bus.soc_kwh)
            if shortage:
                reward += self._charge_cost("bus_battery_violation", shortage)
        unloading = 0.0
        for parcel_id in self.pending_bus_parcels.pop((trip_id, station_id), []):
            parcel = self.parcels[parcel_id]
            unloading += parcel.weight_kg * float(self.config["bus"]["station_unloading_time_min_per_kg"])
            self.bus_freight_kg[trip_id] -= parcel.weight_kg
            self._push(self.now_min + unloading, "parcel_station_arrival", {"parcel_id": parcel_id, "station_id": station_id})
        event.payload["unloading_delay_min"] = unloading
        return reward

    def _bus_mask(self, event: Event) -> list[bool]:
        station = self.stations[event.payload["station_id"]]
        station.active_bus_charges = [end for end in station.active_bus_charges if end > self.now_min + EPSILON]
        charger_available = len(station.active_bus_charges) < int(next(row["charger_num"] for row in self.station_rows if row["station_id"] == station.station_id))
        # Power capacity is a soft constraint: expose charging whenever a physical
        # charger is available, then price any overload in ``_apply_bus_action``.
        return [True] + [charger_available] * (self.bus_action_size - 1)

    def _apply_bus_loading_action(self, event: Event, action: int) -> float:
        trip_id = event.payload["trip_id"]
        surface = build_bus_loading_decision_surface(self, trip_id)
        candidate = surface.candidates[action]
        if candidate.action_type != "load_ready":
            return 0.0
        ready = [pid for q in self.bus_terminal_ready.values() for pid in q]
        capacity = float(self.config["bus"]["freight_capacity_kg"])
        remaining = max(0.0, capacity - self.bus_freight_kg[trip_id])
        loaded: list[str] = []
        total_weight = 0.0
        for parcel_id in list(ready):
            parcel = self.parcels[parcel_id]
            if total_weight + parcel.weight_kg <= remaining + EPSILON:
                loaded.append(parcel_id)
                total_weight += parcel.weight_kg
        for parcel_id in loaded:

            for q in self.bus_terminal_ready.values():
                if parcel_id in q:
                    q.remove(parcel_id); break
            parcel = self.parcels[parcel_id]
            station_id = str(parcel.station_id)
            self.bus_freight_kg[trip_id] += parcel.weight_kg
            self.pending_bus_parcels.setdefault((trip_id, station_id), []).append(parcel_id)
            parcel.status = "ONBOARD_BUS"
        return 0.0

    def _apply_bus_action(self, event: Event, action: int) -> float:
        trip_id, stop_index = event.payload["trip_id"], event.payload["stop_index"]
        station = self.stations[event.payload["station_id"]]
        duration_min = float(self.config["bus"]["charging_actions_sec"][action]) / 60
        unloading = float(event.payload.get("unloading_delay_min", 0.0))
        reward = 0.0
        if duration_min > 0:
            station.active_bus_charges.append(self.now_min + duration_min)
            energy = float(self.config["bus"]["charging_power_kw"]) * duration_min / 60
            bus = self.physical_buses[self.trip_to_bus[trip_id]]
            bus.soc_kwh = min(float(self.config["bus"]["bus_battery_kwh"]), bus.soc_kwh + energy * float(self.config["bus"]["charging_efficiency"]))
            self.bus_soc_kwh[trip_id] = bus.soc_kwh
            reward += self._charge_cost("energy_cost", energy)
            station_load_kw = self._station_load_kw(station, self.now_min)
            self.peak_station_load_kw = max(self.peak_station_load_kw, station_load_kw)
        delay = duration_min + unloading
        bus = self.physical_buses[self.trip_to_bus[trip_id]]
        bus.schedule_delay_min += delay
        self.bus_delay_min[trip_id] = bus.schedule_delay_min
        reward += self._charge_cost("passenger_delay", duration_min)
        reward += self._charge_cost("bus_operating_delay", delay)
        rows = self.trip_stop_times[trip_id]
        next_index = next((i for i in range(stop_index + 1, len(rows)) if rows[i]["stop_id"] in self.stop_to_station), None)
        if next_index is not None:
            scheduled_delta = float(rows[next_index]["arrival_time"]) - float(rows[stop_index]["arrival_time"])
            self._push(self.now_min + delay + max(0.0, scheduled_delta), "bus_arrival", {"trip_id": trip_id, "stop_index": next_index, "previous_stop_index": stop_index})
        else:
            bus = self.physical_buses[self.trip_to_bus[trip_id]]
            relocation_t = float(next((r.get("relocation_time_min", 0.0) for r in self.trip_to_bus_rows if r.get("trip_id") == trip_id), 0.0))
            layover_t = float(next((r.get("minimum_layover_min", 0.0) for r in self.trip_to_bus_rows if r.get("trip_id") == trip_id), 0.0))
            relocation_km = relocation_t / 60.0 * float(self.config["bus"].get("bus_speed_kmph", 25.0))
            bus.last_relocation_energy_kwh = relocation_km * float(self.config["bus"].get("bus_energy_kwh_per_km", 1.6))
            bus.soc_kwh -= bus.last_relocation_energy_kwh
            bus.next_available_time_min = self.now_min + delay + relocation_t + layover_t
            bus.current_trip_id = None
            bus.relocation_status = "layover_complete_scheduled"
            self.bus_soc_kwh[trip_id] = bus.soc_kwh
        return reward

    def _handle_station_arrival(self, parcel_id: str, station_id: str) -> float:
        parcel, station = self.parcels[parcel_id], self.stations[station_id]
        parcel.status = "AT_STATION"
        station.locker_load_kg += parcel.weight_kg
        self.waiting_station_parcels.setdefault(station_id, []).append(parcel_id)
        self._push(self.now_min, "station_operation", {"station_id": station_id})
        return 0.0

    def _apply_station_action(self, event: Event, action: int) -> float:
        station_id = event.payload["station_id"]
        surface = build_station_decision_surface(self, station_id)
        candidate = surface.candidates[action]
        if candidate.action_type != "dispatch_drone":
            return 0.0
        waiting = self.waiting_station_parcels.get(station_id, [])
        if candidate.entity_id not in waiting:
            return self._charge_cost("infeasible_action", 1.0)
        station = self.stations[station_id]
        drone_index = min(range(station.drones), key=station.drone_available_min.__getitem__)
        if station.full_batteries <= 0:
            return self._charge_cost("battery_shortage", 1.0)
        if station.drone_available_min[drone_index] > self.now_min + EPSILON:
            return 0.0
        waiting.remove(candidate.entity_id)
        station.full_batteries -= 1
        return self._dispatch_drone(candidate.entity_id, station_id, drone_index, True)

    def _drone_delivery_times(
        self, parcel_id: str, station_id: str, dispatch: float
    ) -> tuple[float, float]:
        distance_km = float(self.drone_distance_m[self.drone_row_index[station_id], self.drone_column_index[parcel_id]]) / 1000
        one_way = distance_km / float(self.config["network"]["drone_speed_kmph"]) * 60
        delivery = dispatch + one_way + float(self.config["network"]["customer_service_time_min"])
        drone_return = delivery + one_way + float(self.config["network"]["drone_turnaround_time_min"])
        return delivery, drone_return

    def _dispatch_drone(
        self, parcel_id: str, station_id: str, drone_index: int, battery_reserved: bool
    ) -> float:
        parcel, station = self.parcels[parcel_id], self.stations[station_id]
        if not battery_reserved and station.full_batteries > 0:
            station.full_batteries -= 1
        delivery, drone_return = self._drone_delivery_times(parcel_id, station_id, self.now_min)
        station.drone_available_min[drone_index] = drone_return
        station.locker_load_kg = max(0.0, station.locker_load_kg - parcel.weight_kg)
        if self.now_min <= self.horizon_min:
            self._push(drone_return, "drone_return", {"station_id": station_id, "drone_index": drone_index})
            battery_ready = drone_return + station.battery_charge_duration_min
            heapq.heappush(station.battery_ready_min, battery_ready)
            station.active_battery_charges.append((drone_return, battery_ready))
            self._push(battery_ready, "battery_ready", {"station_id": station_id})
        parcel.status = "ONBOARD_DRONE"
        self._push(delivery, "parcel_delivery", {"parcel_id": parcel_id})
        return 0.0

    def _charge_cost(self, component: str, amount: float) -> float:
        weighted = max(0.0, amount) * float(self.config["reward"][component])
        self.cost_components[component] += weighted
        return -weighted

    def _finish_episode(self) -> float:
        if self.terminated:
            return 0.0
        self.now_min = min(max(self.now_min, self.horizon_min), self.horizon_min)
        undelivered = sum(parcel.priority for parcel in self.parcels.values() if parcel.status != "DELIVERED")
        self.terminated = True
        self.current_decision = None
        return self._charge_cost("undelivered", float(undelivered))

    def _decision_surface(self) -> DecisionSurface | None:
        if self.current_decision is None:
            return None
        event = self.current_decision.event
        if self.current_decision.agent == "assignment":
            parcel = self.parcels[event.payload["parcel_id"]]
            return build_assignment_decision_surface(
                self, parcel, list(self.current_decision.action_mask)
            )
        if self.current_decision.agent == "truck":
            truck = self.trucks[int(event.payload["truck_index"])]
            return build_truck_decision_surface(self, truck)
        if self.current_decision.agent == "bus" and event.kind == "bus_departure":
            return build_bus_loading_decision_surface(self, event.payload["trip_id"])
        if self.current_decision.agent == "bus":
            return build_bus_charging_decision_surface(self, event)
        if self.current_decision.agent == "station":
            return build_station_decision_surface(self, event.payload["station_id"])
        raise ValueError(f"Unknown decision agent: {self.current_decision.agent}")

    def _observation(self) -> dict[str, Any]:
        surface = self._decision_surface()
        if surface is None:
            return {
                "agent": "terminal",
                "agent_id": "terminal",
                "event_type": "TERMINAL",
                "entity_id": None,
                "time_min": self.now_min,
                "features": [1.0],
                "feature_names": ("terminal",),
                "action_mask": [],
                "candidate_actions": [],
                "candidate_features": [],
                "candidate_feature_names": (),
            }
        return {
            "agent": surface.agent_id,
            "agent_id": surface.agent_id,
            "event_type": surface.event_type,
            "entity_id": surface.entity_id,
            "time_min": self.now_min,
            "features": list(surface.features),
            "feature_names": list(surface.feature_names),
            "action_mask": surface.action_mask(),
            "candidate_actions": surface.candidate_payloads(),
            "candidate_features": surface.candidate_feature_matrix(),
            "candidate_feature_names": list(surface.candidate_feature_names()),
        }

    def get_global_state(self) -> list[float]:
        """Return a fixed-size centralized state shared by both Stage 7 agents."""
        parcel_count = max(len(self.parcels), 1)
        station_count = max(len(self.stations), 1)
        trip_count = max(len(self.bus_soc_kwh), 1)
        delivered = sum(parcel.status == "DELIVERED" for parcel in self.parcels.values())
        awaiting = sum(parcel.status == "PENDING_ASSIGNMENT" for parcel in self.parcels.values())
        in_transit = sum(parcel.status == "ONBOARD_TRUCK" for parcel in self.parcels.values())
        battery_capacity = max(float(self.config["bus"]["bus_battery_kwh"]), 1.0)
        locker_ratios = [station.locker_load_kg / max(station.locker_capacity_kg, 1.0) for station in self.stations.values()]
        battery_counts = [station.full_batteries / max(float(self.config["station"]["initial_full_batteries"]), 1.0) for station in self.stations.values()]
        return [
            self.now_min / max(self.horizon_min, 1.0),
            delivered / parcel_count, awaiting / parcel_count, in_transit / parcel_count,
            self.decision_counts["assignment"] / parcel_count,
            self.decision_counts["bus"] / max(trip_count * station_count, 1),
            sum(self.bus_soc_kwh.values()) / (trip_count * battery_capacity),
            sum(self.bus_delay_min.values()) / (trip_count * max(self.horizon_min, 1.0)),
            sum(self.bus_freight_kg.values()) / (trip_count * max(float(self.config["bus"]["freight_capacity_kg"]), 1.0)),
            sum(locker_ratios) / station_count,
            sum(battery_counts) / station_count,
            sum(self.truck_available_min) / (max(len(self.truck_available_min), 1) * max(self.horizon_min, 1.0)),
            self.infeasible_action_corrections / max(sum(self.decision_counts.values()), 1),
            float(self.terminated), float(self.truncated),
        ]

    def _info(self, step_reward: float) -> dict[str, Any]:
        delivered = sum(parcel.status == "DELIVERED" for parcel in self.parcels.values())
        drone_deliveries = sum(
            parcel.status == "DELIVERED" and parcel.mode in {"TBD", "TLD"}
            for parcel in self.parcels.values()
        )
        metrics = {
            "decision_events": sum(self.decision_counts.values()),
            "assignment_events": self.decision_counts["assignment"],
            "bus_charging_events": self.decision_counts["bus"],
            "delivered_parcels": delivered,
            "undelivered_parcels": len(self.parcels) - delivered,
            "drone_deliveries": drone_deliveries,
            "total_reward": self.reward_total,
            "infeasible_action_corrections": self.infeasible_action_corrections,
            "fallback_feasibility_events": self.fallback_feasibility_events,
            "power_overload_amount": self.accumulated_power_overload,
            "power_overload_duration": self.accumulated_power_overload_duration,
            "locker_overflow_amount": self.accumulated_locker_overflow,
            "locker_overflow_duration": self.accumulated_locker_overflow_duration,
            "truck_total_distance": sum(truck.total_distance for truck in self.trucks),
            "truck_dispatch_count": self.truck_dispatch_count,
            "average_weight_utilization": self.truck_weight_utilization_sum / max(self.truck_dispatch_count, 1),
            "average_volume_utilization": self.truck_volume_utilization_sum / max(self.truck_dispatch_count, 1),
            "average_parcels_per_route": self.truck_parcels_routed / max(self.truck_dispatch_count, 1),
            "truck_operating_cost": (
                self.cost_components["truck_cost"]
                / max(float(self.config["reward"]["truck_cost"]), EPSILON)
            ),
        }
        return {
            "time_min": self.now_min,
            "decision_agent": self.current_decision.agent if self.current_decision else None,
            "step_reward": step_reward,
            "episode_reward": self.reward_total,
            "reward_components": {name: -value for name, value in self.cost_components.items()},
            "cost_components": dict(self.cost_components),
            "metrics": metrics,
            "delivered_parcels": delivered,
            "total_parcels": len(self.parcels),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return the current dependency-light environment metric snapshot."""
        return dict(self._info(0.0)["metrics"])

    def check_invariants(self) -> list[str]:
        """Return invariant violations; an empty list means the state is valid."""
        errors = []
        if self.now_min < -EPSILON or self.now_min > self.horizon_min + EPSILON:
            errors.append("simulation time is outside the delivery horizon")
        for station in self.stations.values():
            if station.locker_load_kg < -EPSILON:
                errors.append(f"{station.station_id} has negative locker load")
            if station.full_batteries < 0:
                errors.append(f"{station.station_id} has negative full-battery count")
            if len(station.drone_available_min) != station.drones:
                errors.append(f"{station.station_id} drone availability vector has the wrong size")
        if any(value < -EPSILON for value in self.bus_freight_kg.values()):
            errors.append("a bus has negative freight load")
        valid_truck_statuses = {"idle", "traveling", "loading", "unloading"}
        for truck in self.trucks:
            if truck.available_time < -EPSILON:
                errors.append(f"{truck.truck_id} has negative availability time")
            if truck.remaining_capacity_kg < -EPSILON:
                errors.append(f"{truck.truck_id} has negative remaining capacity")
            if truck.status not in valid_truck_statuses:
                errors.append(f"{truck.truck_id} has invalid status {truck.status}")
        return errors


def first_feasible_policy(observation: dict[str, Any]) -> int:
    """Deterministic baseline used by the offline smoke test."""
    return next(index for index, feasible in enumerate(observation["action_mask"]) if feasible)
