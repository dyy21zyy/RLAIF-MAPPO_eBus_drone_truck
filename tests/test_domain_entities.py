from dataclasses import fields

from envs.entities import (
    BatteryState,
    PhysicalBusState,
    ScheduledTrip,
    StationState,
    TruckState,
    ParcelState,
)


def test_required_parcel_and_battery_statuses():
    assert {s.name for s in ParcelState} >= {
        "UNRELEASED","PENDING_ASSIGNMENT","WAITING_TRUCK","ONBOARD_TRUCK",
        "AT_BUS_TERMINAL","ONBOARD_BUS","AT_STATION","WAITING_DRONE",
        "ONBOARD_DRONE","DELIVERED","FAILED",
    }
    assert {s.name for s in BatteryState} >= {"FULL", "IN_USE", "DEPLETED", "CHARGING"}


def test_physical_bus_state_distinct_from_scheduled_trip():
    assert PhysicalBusState is not ScheduledTrip
    assert "physical_bus_id" in {f.name for f in fields(PhysicalBusState)}
    assert "trip_id" in {f.name for f in fields(ScheduledTrip)}


def test_truck_state_has_weight_and_volume_capacity():
    names = {f.name for f in fields(TruckState)}
    assert {"weight_capacity_kg", "volume_capacity_m3"} <= names


def test_station_state_has_charging_slot_capacity():
    assert "charging_slots" in {f.name for f in fields(StationState)}
