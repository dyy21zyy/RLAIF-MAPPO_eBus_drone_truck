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

from envs.state_builder import build_assignment_features
from utils.config import load_config

EPSILON = 1e-9
EVENT_PRIORITY = {
    "battery_ready": 0,
    "drone_return": 1,
    "parcel_delivery": 2,
    "parcel_station_arrival": 3,
    "bus_arrival": 4,
    "parcel_release": 5,
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
    priority: int
    nearest_station_id: str
    drone_feasible: bool
    status: str = "unreleased"
    action_id: int | None = None
    mode: str | None = None
    station_id: str | None = None
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
    drone_available_min: list[float] = field(default_factory=list)
    battery_ready_min: list[float] = field(default_factory=list)


@dataclass
class Decision:
    agent: str
    event: Event
    action_mask: list[bool]


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
        if not parcel_rows or not station_rows or not trip_rows or not stop_time_rows:
            raise InstanceValidationError("Parcels, stations, trips, and stop times must be non-empty")

        self.parcel_rows = sorted(parcel_rows, key=lambda row: (float(row["release_time"]), row["parcel_id"]))
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
        self.trip_stop_times: dict[str, list[dict[str, str]]] = {}
        for row in stop_time_rows:
            self.trip_stop_times.setdefault(row["trip_id"], []).append(row)
        for rows in self.trip_stop_times.values():
            rows.sort(key=lambda row: int(row["stop_sequence"]))
        missing_trips = set(self.trip_rows) - set(self.trip_stop_times)
        if missing_trips:
            raise InstanceValidationError(f"Trips have no stop times: {sorted(missing_trips)}")

    @property
    def assignment_action_size(self) -> int:
        return 1 + 2 * len(self.station_ids)

    @property
    def bus_action_size(self) -> int:
        return len(self.config["bus"]["charging_actions_sec"])

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
        self.truck_available_min = [0.0] * int(self.config["truck"]["num_trucks"])
        self.bus_soc_kwh: dict[str, float] = {trip_id: float(self.config["bus"]["bus_battery_kwh"]) for trip_id in self.trip_rows}
        self.bus_delay_min: dict[str, float] = {trip_id: 0.0 for trip_id in self.trip_rows}
        self.bus_freight_kg: dict[str, float] = {trip_id: 0.0 for trip_id in self.trip_rows}
        self.bus_segment_index: dict[str, int] = {trip_id: 0 for trip_id in self.trip_rows}
        self.pending_bus_parcels: dict[tuple[str, str], list[str]] = {}
        self.reward_total = 0.0
        self.decision_counts = {"assignment": 0, "bus": 0}
        self.infeasible_action_corrections = 0
        self.cost_components = {name: 0.0 for name in (
            "passenger_delay", "bus_operating_delay", "parcel_lateness", "energy_cost", "power_overload",
            "bus_battery_violation", "locker_overflow", "truck_cost", "undelivered", "battery_shortage", "infeasible_action")}
        self.parcels = {
            row["parcel_id"]: ParcelState(
                parcel_id=row["parcel_id"], release_time_min=float(row["release_time"]),
                deadline_min=float(row["deadline"]), weight_kg=float(row["weight"]),
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
        if decision.agent == "assignment":
            reward += self._apply_assignment(decision.event.payload["parcel_id"], selected)
        else:
            reward += self._apply_bus_action(decision.event, selected)
        reward += self._advance()
        self.reward_total += reward
        info = self._info(reward)
        info.update({"requested_action": int(action), "applied_action": selected, "action_corrected": corrected})
        return self._observation(), reward, self.terminated, self.truncated, info

    def _advance(self) -> float:
        reward = 0.0
        while self.events and self.current_decision is None:
            event = heapq.heappop(self.events)
            if event.time_min > self.horizon_min + EPSILON:
                self.now_min = self.horizon_min
                break
            self.now_min = max(self.now_min, event.time_min)
            if event.kind == "parcel_release":
                parcel = self.parcels[event.payload["parcel_id"]]
                parcel.status = "awaiting_assignment"
                self.current_decision = Decision("assignment", event, self._assignment_mask(parcel))
            elif event.kind == "bus_arrival":
                reward += self._handle_bus_arrival(event)
                self.current_decision = Decision("bus", event, self._bus_mask(event))
            elif event.kind == "parcel_delivery":
                parcel = self.parcels[event.payload["parcel_id"]]
                parcel.status, parcel.delivered_time_min = "delivered", self.now_min
                lateness = max(0.0, self.now_min - parcel.deadline_min) * parcel.priority
                parcel.cost += lateness
                reward += self._charge_cost("parcel_lateness", lateness)
            elif event.kind == "parcel_station_arrival":
                reward += self._handle_station_arrival(event.payload["parcel_id"], event.payload["station_id"])
            elif event.kind == "drone_return":
                station = self.stations[event.payload["station_id"]]
                station.drone_available_min[event.payload["drone_index"]] = self.now_min
            elif event.kind == "battery_ready":
                station = self.stations[event.payload["station_id"]]
                station.full_batteries += 1
                if station.battery_ready_min:
                    heapq.heappop(station.battery_ready_min)
        if self.current_decision is None and (not self.events or self.now_min >= self.horizon_min - EPSILON):
            reward += self._finish_episode()
        return reward

    def _assignment_mask(self, parcel: ParcelState) -> list[bool]:
        mask = [parcel.weight_kg <= float(self.config["truck"]["capacity_kg"])]
        for station_id in self.station_ids:
            mask.append(parcel.drone_feasible and parcel.weight_kg <= float(self.config["bus"]["freight_capacity_kg"]) and self._next_freight_trip(parcel.release_time_min, station_id, parcel.weight_kg) is not None)
        for station_id in self.station_ids:
            mask.append(parcel.drone_feasible and parcel.weight_kg <= float(self.config["truck"]["capacity_kg"]))
        if not any(mask):
            mask[0] = True  # The correction path can still record an eventual undelivered parcel.
        return mask

    def _next_freight_trip(self, ready_min: float, station_id: str, weight_kg: float) -> tuple[str, float] | None:
        station_stop = self.stations[station_id].stop_id
        loading = weight_kg * float(self.config["bus"]["terminal_loading_time_min_per_kg"])
        candidates = []
        for trip_id, trip in self.trip_rows.items():
            if not self._as_bool(trip["freight_allowed"]):
                continue
            rows = self.trip_stop_times[trip_id]
            terminal_time = float(rows[0]["departure_time"])
            target = next((row for row in rows if row["stop_id"] == station_stop), None)
            if target and ready_min + loading <= terminal_time + EPSILON and self.bus_freight_kg[trip_id] + weight_kg <= float(self.config["bus"]["freight_capacity_kg"]) + EPSILON:
                candidates.append((float(target["arrival_time"]), trip_id))
        candidates.sort()
        return (candidates[0][1], candidates[0][0]) if candidates else None

    def _apply_assignment(self, parcel_id: str, action: int) -> float:
        parcel = self.parcels[parcel_id]
        parcel.action_id = action
        if action == 0:
            parcel.mode = "TD"
            truck = min(range(len(self.truck_available_min)), key=self.truck_available_min.__getitem__)
            start = max(self.now_min, self.truck_available_min[truck]) + parcel.weight_kg * float(self.config["truck"]["loading_time_min_per_kg"])
            depot = self.truck_location_index["depot_01"]
            customer = self.truck_location_index[parcel_id]
            travel = float(self.truck_time_min[depot, customer])
            outbound_distance_km = float(self.truck_distance_m[depot, customer]) / 1000
            return_distance_km = float(self.truck_distance_m[customer, depot]) / 1000 if self.config["truck"]["return_to_depot"] else 0.0
            distance_km = outbound_distance_km + return_distance_km
            service = float(self.config["network"]["customer_service_time_min"])
            return_time = float(self.truck_time_min[customer, depot]) if self.config["truck"]["return_to_depot"] else 0.0
            completion = start + travel + service
            self.truck_available_min[truck] = completion + return_time
            parcel.status = "in_transit"
            self._push(completion, "parcel_delivery", {"parcel_id": parcel_id})
            truck_cost = float(self.config["truck"]["fixed_dispatch_cost"]) + distance_km * float(self.config["truck"]["cost_per_km"]) + (travel + return_time) * float(self.config["truck"]["cost_per_min"])
            return self._charge_cost("truck_cost", truck_cost)
        station_offset = action - 1
        if station_offset < len(self.station_ids):
            parcel.mode = "TBD"
            station_id = self.station_ids[station_offset]
            trip = self._next_freight_trip(self.now_min, station_id, parcel.weight_kg)
            if trip is None:  # Defensive; masks normally prevent this.
                parcel.status = "undelivered"
                return self._charge_cost("infeasible_action", 1.0)
            trip_id, arrival = trip
            self.bus_freight_kg[trip_id] += parcel.weight_kg
            self.pending_bus_parcels.setdefault((trip_id, station_id), []).append(parcel_id)
            parcel.status, parcel.station_id = "on_bus", station_id
            return 0.0
        parcel.mode = "TLD"
        station_id = self.station_ids[station_offset - len(self.station_ids)]
        parcel.station_id = station_id
        truck = min(range(len(self.truck_available_min)), key=self.truck_available_min.__getitem__)
        start = max(self.now_min, self.truck_available_min[truck]) + parcel.weight_kg * float(self.config["truck"]["loading_time_min_per_kg"])
        depot, station = self.truck_location_index["depot_01"], self.truck_location_index[station_id]
        travel = float(self.truck_time_min[depot, station])
        return_time = float(self.truck_time_min[station, depot]) if self.config["truck"]["return_to_depot"] else 0.0
        arrival = start + travel + parcel.weight_kg * float(self.config["truck"]["unloading_time_min_per_kg"])
        self.truck_available_min[truck] = arrival + return_time
        parcel.status = "to_station"
        self._push(arrival, "parcel_station_arrival", {"parcel_id": parcel_id, "station_id": station_id})
        outbound_distance_km = float(self.truck_distance_m[depot, station]) / 1000
        return_distance_km = float(self.truck_distance_m[station, depot]) / 1000 if self.config["truck"]["return_to_depot"] else 0.0
        distance_km = outbound_distance_km + return_distance_km
        truck_cost = float(self.config["truck"]["fixed_dispatch_cost"]) + distance_km * float(self.config["truck"]["cost_per_km"]) + (travel + return_time) * float(self.config["truck"]["cost_per_min"])
        return self._charge_cost("truck_cost", truck_cost)

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
            self.bus_soc_kwh[trip_id] -= segment_km * float(self.config["bus"]["bus_energy_kwh_per_km"])
            shortage = max(0.0, float(self.config["bus"]["bus_min_soc_kwh"]) - self.bus_soc_kwh[trip_id])
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

    def _apply_bus_action(self, event: Event, action: int) -> float:
        trip_id, stop_index = event.payload["trip_id"], event.payload["stop_index"]
        station = self.stations[event.payload["station_id"]]
        duration_min = float(self.config["bus"]["charging_actions_sec"][action]) / 60
        unloading = float(event.payload.get("unloading_delay_min", 0.0))
        reward = 0.0
        if duration_min > 0:
            station.active_bus_charges.append(self.now_min + duration_min)
            energy = float(self.config["bus"]["charging_power_kw"]) * duration_min / 60
            self.bus_soc_kwh[trip_id] = min(float(self.config["bus"]["bus_battery_kwh"]), self.bus_soc_kwh[trip_id] + energy * float(self.config["bus"]["charging_efficiency"]))
            reward += self._charge_cost("energy_cost", energy)
            base_load = float(self.config["station"]["base_load_kw"])
            battery_load = len(station.battery_ready_min) * station.battery_power_kw
            concurrent_bus_load = len(station.active_bus_charges) * float(self.config["bus"]["charging_power_kw"])
            overload_kw = max(0.0, base_load + battery_load + concurrent_bus_load - station.power_capacity_kw)
            reward += self._charge_cost("power_overload", overload_kw * duration_min / 60.0)
        delay = duration_min + unloading
        self.bus_delay_min[trip_id] += delay
        reward += self._charge_cost("passenger_delay", duration_min)
        reward += self._charge_cost("bus_operating_delay", delay)
        rows = self.trip_stop_times[trip_id]
        next_index = next((i for i in range(stop_index + 1, len(rows)) if rows[i]["stop_id"] in self.stop_to_station), None)
        if next_index is not None:
            scheduled_delta = float(rows[next_index]["arrival_time"]) - float(rows[stop_index]["arrival_time"])
            self._push(self.now_min + delay + max(0.0, scheduled_delta), "bus_arrival", {"trip_id": trip_id, "stop_index": next_index, "previous_stop_index": stop_index})
        return reward

    def _handle_station_arrival(self, parcel_id: str, station_id: str) -> float:
        parcel, station = self.parcels[parcel_id], self.stations[station_id]
        parcel.status = "at_station"
        station.locker_load_kg += parcel.weight_kg
        reward = 0.0
        overflow = max(0.0, station.locker_load_kg - station.locker_capacity_kg)
        if overflow:
            reward += self._charge_cost("locker_overflow", overflow)
        drone_index = min(range(station.drones), key=station.drone_available_min.__getitem__)
        dispatch = max(self.now_min, station.drone_available_min[drone_index])
        if station.full_batteries <= 0:
            reward += self._charge_cost("battery_shortage", 1.0)
            dispatch = max(dispatch, station.battery_ready_min[0] if station.battery_ready_min else self.horizon_min + 1)
        else:
            station.full_batteries -= 1
        distance_km = float(self.drone_distance_m[self.drone_row_index[station_id], self.drone_column_index[parcel_id]]) / 1000
        one_way = distance_km / float(self.config["network"]["drone_speed_kmph"]) * 60
        delivery = dispatch + one_way + float(self.config["network"]["customer_service_time_min"])
        drone_return = delivery + one_way + float(self.config["network"]["drone_turnaround_time_min"])
        station.drone_available_min[drone_index] = drone_return
        station.locker_load_kg = max(0.0, station.locker_load_kg - parcel.weight_kg)
        if dispatch <= self.horizon_min:
            self._push(drone_return, "drone_return", {"station_id": station_id, "drone_index": drone_index})
            battery_ready = drone_return + station.battery_charge_duration_min
            heapq.heappush(station.battery_ready_min, battery_ready)
            self._push(battery_ready, "battery_ready", {"station_id": station_id})
        parcel.status = "out_for_delivery"
        self._push(delivery, "parcel_delivery", {"parcel_id": parcel_id})
        return reward

    def _charge_cost(self, component: str, amount: float) -> float:
        weighted = max(0.0, amount) * float(self.config["reward"][component])
        self.cost_components[component] += weighted
        return -weighted

    def _finish_episode(self) -> float:
        if self.terminated:
            return 0.0
        self.now_min = min(max(self.now_min, self.horizon_min), self.horizon_min)
        undelivered = sum(parcel.priority for parcel in self.parcels.values() if parcel.status != "delivered")
        self.terminated = True
        self.current_decision = None
        return self._charge_cost("undelivered", float(undelivered))

    def _observation(self) -> dict[str, Any]:
        if self.current_decision is None:
            return {"agent": "terminal", "features": [1.0], "action_mask": []}
        event = self.current_decision.event
        if self.current_decision.agent == "assignment":
            parcel = self.parcels[event.payload["parcel_id"]]
            features = build_assignment_features(self, parcel)
            entity_id = parcel.parcel_id
        else:
            trip_id, station_id = event.payload["trip_id"], event.payload["station_id"]
            station = self.stations[station_id]
            features = [
                self.now_min / max(self.horizon_min, 1.0),
                self.bus_soc_kwh[trip_id] / max(float(self.config["bus"]["bus_battery_kwh"]), 1.0),
                self.bus_delay_min[trip_id] / max(self.horizon_min, 1.0),
                station.locker_load_kg / max(station.locker_capacity_kg, 1.0),
                station.full_batteries / max(float(self.config["station"]["initial_full_batteries"]), 1.0),
                self.bus_freight_kg[trip_id] / max(float(self.config["bus"]["freight_capacity_kg"]), 1.0),
            ]
            entity_id = f"{trip_id}:{station_id}"
        return {"agent": self.current_decision.agent, "entity_id": entity_id, "time_min": self.now_min,
                "features": features, "action_mask": list(self.current_decision.action_mask)}

    def _info(self, step_reward: float) -> dict[str, Any]:
        delivered = sum(parcel.status == "delivered" for parcel in self.parcels.values())
        drone_deliveries = sum(
            parcel.status == "delivered" and parcel.mode in {"TBD", "TLD"}
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
        if any(value < -EPSILON for value in self.truck_available_min):
            errors.append("a truck has negative availability time")
        return errors


def first_feasible_policy(observation: dict[str, Any]) -> int:
    """Deterministic baseline used by the offline smoke test."""
    return next(index for index, feasible in enumerate(observation["action_mask"]) if feasible)
