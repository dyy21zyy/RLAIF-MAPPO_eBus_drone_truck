"""Typed Phase 0 entities for the final dynamic multi-agent contract.

These dataclasses define the paper-code interface needed by later phases.  They
are intentionally lightweight schema objects and are not runtime implementations
of batching, physical-bus circulation, passenger dynamics, station operations,
or multi-agent RLAIF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from envs.event_types import EventType

RLAIF_AGENT_TYPES = {"assignment", "truck", "bus", "station"}


class ParcelState(str, Enum):
    UNRELEASED = "UNRELEASED"
    PENDING_ASSIGNMENT = "PENDING_ASSIGNMENT"
    WAITING_TRUCK = "WAITING_TRUCK"
    ONBOARD_TRUCK = "ONBOARD_TRUCK"
    AT_BUS_TERMINAL = "AT_BUS_TERMINAL"
    ONBOARD_BUS = "ONBOARD_BUS"
    AT_STATION = "AT_STATION"
    WAITING_DRONE = "WAITING_DRONE"
    ONBOARD_DRONE = "ONBOARD_DRONE"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


class DroneStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    IN_MISSION = "IN_MISSION"
    RETURNING = "RETURNING"
    TURNAROUND = "TURNAROUND"


class BatteryState(str, Enum):
    FULL = "FULL"
    IN_USE = "IN_USE"
    DEPLETED = "DEPLETED"
    CHARGING = "CHARGING"


@dataclass(frozen=True)
class ParcelAssignment:
    parcel_id: str
    assignment_action: str
    delivery_mode: str
    target_station_id: str | None = None
    assigned_at_min: float = 0.0
    terminal_transfer_required: bool = False

    def __post_init__(self) -> None:
        if self.delivery_mode == "TBD" and self.target_station_id is None:
            raise ValueError("TBD assignments must carry only a downstream target station")


@dataclass(frozen=True)
class TruckState:
    truck_id: str
    location_id: str
    available_at_min: float
    weight_capacity_kg: float
    volume_capacity_m3: float
    onboard_parcel_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TruckRouteStop:
    stop_id: str
    stop_type: str
    parcel_ids: tuple[str, ...]
    service_time_min: float = 0.0


@dataclass(frozen=True)
class TruckRoutePlan:
    route_id: str
    truck_id: str
    stops: tuple[TruckRouteStop, ...]
    total_distance_km: float
    return_to_depot: bool = True


@dataclass(frozen=True)
class TruckBatchCandidate:
    candidate_id: str
    truck_id: str
    parcel_ids: tuple[str, ...]
    ordered_route_stops: tuple[TruckRouteStop, ...]
    total_weight_kg: float
    total_volume_m3: float
    weight_utilization: float
    volume_utilization: float
    estimated_distance_km: float
    estimated_travel_time_min: float
    estimated_loading_time_min: float
    estimated_unloading_time_min: float
    estimated_completion_time_min: float
    estimated_lateness_min: float
    remaining_weight_capacity_kg: float
    remaining_volume_capacity_m3: float
    number_of_direct_customers: int
    number_of_terminal_deliveries: int
    number_of_station_deliveries: int
    feasible: bool
    infeasibility_reasons: tuple[str, ...] = ()
    heuristic_source: str = "unknown"
    idle_flag: bool = False


@dataclass(frozen=True)
class ScheduledTrip:
    trip_id: str
    route_id: str
    origin_stop_id: str
    destination_stop_id: str
    planned_departure_min: float
    planned_arrival_min: float
    freight_enabled: bool = False


@dataclass(frozen=True)
class PassengerManifest:
    onboard_passengers: int = 0
    onboard_passengers_by_destination: dict[str, int] = field(default_factory=dict)
    total_onboard_passengers: int = 0
    passenger_capacity: int = 80
    origin_destination_counts: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalBusState:
    physical_bus_id: str
    current_trip_id: str | None
    location_id: str
    available_at_min: float
    energy_kwh: float
    current_location: str | None = None
    next_available_time_min: float | None = None
    soc_kwh: float | None = None
    schedule_delay_min: float = 0.0
    delay_min: float = 0.0
    manifest: PassengerManifest = field(default_factory=PassengerManifest)
    onboard_parcel_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BusLoadingBatchCandidate:
    candidate_id: str
    physical_bus_id: str
    trip_id: str
    parcel_ids: tuple[str, ...]
    target_station_ids: tuple[str, ...]
    total_weight_kg: float
    freight_capacity_utilization: float
    loading_time_min: float
    unload_weight_by_station: dict[str, float]
    estimated_unloading_time_by_station: dict[str, float]
    maximum_single_station_unload_kg: float
    estimated_lateness_min: float
    estimated_passenger_impact_min: float
    feasible: bool
    infeasibility_reasons: tuple[str, ...] = ()
    heuristic_source: str = "unknown"
    idle_flag: bool = False


@dataclass(frozen=True)
class BusChargingDurationCandidate:
    candidate_id: str
    physical_bus_id: str
    station_id: str
    duration_sec: int
    energy_added_kwh: float
    projected_soc_kwh: float
    projected_overload_kw: float
    feasible: bool
    infeasibility_reasons: tuple[str, ...] = ()
    idle_flag: bool = False


@dataclass(frozen=True)
class PassengerStopState:
    stop_id: str
    waiting_by_destination: dict[str, int] = field(default_factory=dict)
    total_waiting: int = 0
    last_queue_update_time: float = 0.0
    cumulative_waiting_passenger_minutes: float = 0.0
    waiting_passengers: int = 0
    waiting_passenger_minutes: float = 0.0
    onboard_additional_delay_passenger_minutes: float = 0.0


@dataclass(frozen=True)
class DroneState:
    drone_id: str
    home_station_id: str
    status: str = DroneStatus.AVAILABLE.value
    available_time_min: float = 0.0
    active_parcel_id: str | None = None
    active_battery_id: str | None = None

    @property
    def station_id(self) -> str:
        return self.home_station_id

    @property
    def available_at_min(self) -> float:
        return self.available_time_min


@dataclass(frozen=True)
class BatteryStateEntity:
    battery_id: str
    home_station_id: str
    status: str = BatteryState.FULL.value
    charge_start_time_min: float | None = None
    charge_completion_time_min: float | None = None
    assigned_drone_id: str | None = None


@dataclass(frozen=True)
class BatteryChargingJob:
    battery_id: str
    station_id: str
    start_time_min: float
    end_time_min: float
    power_kw: float


@dataclass(frozen=True)
class StationState:
    station_id: str
    locker_capacity_kg: float
    current_locker_load_kg: float
    charging_slots: int
    power_capacity_kw: float
    full_battery_ids: tuple[str, ...] = ()
    depleted_battery_ids: tuple[str, ...] = ()
    charging_jobs: tuple[BatteryChargingJob, ...] = ()


@dataclass(frozen=True)
class StationOperationCandidate:
    candidate_id: str
    station_id: str
    dispatches: tuple[tuple[str, str, str], ...]
    battery_ids_to_start_charging: tuple[str, ...]
    estimated_delivery_times: tuple[float, ...] = ()
    estimated_return_times: tuple[float, ...] = ()
    estimated_parcel_lateness: tuple[float, ...] = ()
    full_batteries_remaining: int = 0
    depleted_batteries_remaining: int = 0
    available_drones_remaining: int = 0
    charging_slots_used_after_action: int = 0
    projected_station_load: float = 0.0
    projected_overload: float = 0.0
    power_margin: float = 0.0
    expected_overload_duration: float = 0.0
    feasible: bool = True
    infeasibility_reasons: tuple[str, ...] = ()
    heuristic_source: str = "unknown"
    idle_flag: bool = False

    @property
    def drone_parcel_matches(self) -> tuple[tuple[str, str], ...]:
        return tuple((d, p) for d, p, _ in self.dispatches)

    @property
    def batteries_to_start_charging(self) -> tuple[str, ...]:
        return self.battery_ids_to_start_charging

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.infeasibility_reasons


@dataclass(frozen=True)
class DecisionEvent:
    event_id: str
    event_type: EventType
    time_min: float
    agent_type: str
    entity_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardLedgerEntry:
    event_id: str
    agent_type: str
    reward_components: dict[str, float]
    total_reward: float
