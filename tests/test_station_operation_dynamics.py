from types import SimpleNamespace

import pytest

from envs.dynamics.station_dynamics import dispatch_drone, start_battery_charging
from tests.test_station_dispatch_matching import env
from tests.test_four_agent_environment import make_env
from envs.delivery_env import DynamicDeliveryEnv


def _passenger_service_env(stop_ids):
    manifest=SimpleNamespace(
        total_onboard_passengers=0,
        onboard_passengers_by_destination={},
        onboard_additional_delay_passenger_minutes=0.0,
    )
    return SimpleNamespace(
        trip_stop_times={"trip": [{"stop_id": stop_id} for stop_id in stop_ids]},
        physical_buses={"bus": SimpleNamespace(passenger_manifest=manifest)},
        trip_to_bus={"trip": "bus"}, passenger_stops={}, passenger_arrivals=object(),
        now_min=0.0, config={"passenger": {}}, stop_to_station={"station": "s"},
        integrated_stations_visited=0, ordinary_stops_visited=0,
        passenger_boardings_at_ordinary_stops=0,
        passenger_alightings_at_ordinary_stops=0,
        total_passenger_boardings_all_stops=0,
    )


def _stop_result(boardings=0, alightings=0):
    return SimpleNamespace(boarding_count=boardings, alighting_count=alightings,
                           onboard_after_departure=0)


def test_all_stop_boarding_counter_distinguishes_station_and_ordinary(monkeypatch):
    results=iter((_stop_result(2), _stop_result(3), _stop_result(0, 4),
                  _stop_result(0)))
    monkeypatch.setattr("envs.delivery_env.process_bus_stop",
                        lambda *args, **kwargs: next(results))
    test_env=_passenger_service_env(("ordinary", "station", "ordinary", "station"))
    for index in range(4):
        DynamicDeliveryEnv._process_stop_service(
            test_env, "trip", index, incoming_energy=0.0, soc_before=0.0)
    assert test_env.total_passenger_boardings_all_stops == 5
    assert test_env.passenger_boardings_at_ordinary_stops == 2


def test_reset_counts_arrivals_and_clears_all_stop_boardings(tmp_path):
    test_env=make_env(tmp_path)
    for index,row in enumerate(test_env.passenger_rows):
        row["passenger_count"]=f"{index + 1}.0"
    expected=sum(range(1, len(test_env.passenger_rows) + 1))
    test_env.total_passenger_boardings_all_stops=99
    test_env.reset(seed=7)
    assert test_env.total_passenger_arrivals == expected
    assert test_env.total_passenger_boardings_all_stops == 0


def dispatch_parts(test_env, index=0):
    station=test_env.stations["s"]
    return station, station.drone_states[index], test_env.parcels[f"p{index}"], station.battery_states[index]


def test_one_valid_dispatch_increments_mission_count():
    test_env=env()
    dispatch_drone(test_env, *dispatch_parts(test_env), 0.0)
    assert test_env.drone_mission_count == 1


def test_two_valid_dispatches_increment_mission_count_twice():
    test_env=env()
    dispatch_drone(test_env, *dispatch_parts(test_env, 0), 0.0)
    dispatch_drone(test_env, *dispatch_parts(test_env, 1), 0.0)
    assert test_env.drone_mission_count == 2


@pytest.mark.parametrize(("part","message"), (("drone","drone_unavailable"),
                                                ("parcel","parcel_unavailable"),
                                                ("battery","battery_not_full")))
def test_invalid_dispatch_does_not_increment(part, message):
    test_env=env()
    station,drone,parcel,battery=dispatch_parts(test_env)
    if part=="drone": drone.status="IN_MISSION"
    elif part=="parcel": parcel.status="DELIVERED"
    else: battery.status="DEPLETED"
    with pytest.raises(ValueError, match=message):
        dispatch_drone(test_env,station,drone,parcel,battery,0.0)
    assert getattr(test_env,"drone_mission_count",0) == 0


def test_idle_and_battery_charging_do_not_increment_missions():
    test_env=env()
    # Idle performs no station operation. Charging is separately instrumented.
    assert getattr(test_env,"drone_mission_count",0) == 0
    station=test_env.stations["s"]
    battery=station.battery_states[0]
    battery.status="DEPLETED"
    start_battery_charging(test_env,station,battery,0.0)
    assert getattr(test_env,"drone_mission_count",0) == 0


def test_delivery_and_return_do_not_increment_mission_again():
    test_env=env()
    station,drone,parcel,battery=dispatch_parts(test_env)
    dispatch_drone(test_env,station,drone,parcel,battery,0.0)
    parcel.status="DELIVERED"
    drone.status="AVAILABLE"
    battery.status="DEPLETED"
    assert test_env.drone_mission_count == 1


def test_environment_reset_initializes_mission_count(tmp_path):
    test_env=make_env(tmp_path)
    test_env.drone_mission_count=9
    test_env.reset(seed=7)
    assert test_env.drone_mission_count == 0
