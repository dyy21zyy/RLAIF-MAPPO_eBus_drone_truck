from types import SimpleNamespace

import pytest

from envs.dynamics.station_dynamics import dispatch_drone, start_battery_charging
from tests.test_station_dispatch_matching import env
from tests.test_four_agent_environment import make_env


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
