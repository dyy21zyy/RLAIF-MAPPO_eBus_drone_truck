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
    origin_destination_counts: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalBusState:
    physical_bus_id: str
    current_trip_id: str | None
    location_id: str
    available_at_min: float
    energy_kwh: float
    delay_min: float = 0.0
    manifest: PassengerManifest = field(default_factory=PassengerManifest)
    onboard_parcel_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PassengerStopState:
    stop_id: str
    waiting_passengers: int
    waiting_passenger_minutes: float
    onboard_additional_delay_passenger_minutes: float = 0.0


@dataclass(frozen=True)
class DroneState:
    drone_id: str
    station_id: str
    available_at_min: float
    battery_id: str | None = None
    payload_parcel_id: str | None = None


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
    drone_parcel_matches: tuple[tuple[str, str], ...]
    batteries_to_start_charging: tuple[str, ...]
    feasible: bool
    reasons: tuple[str, ...] = ()


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
