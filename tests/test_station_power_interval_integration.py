import pytest
from envs.dynamics.station_power import StationBaseLoadInterval, StationBaseLoadProfile, integrate_station_power

def test_boundary_splitting_scenario():
    p=StationBaseLoadProfile([StationBaseLoadInterval('h','i0',0,15,100),StationBaseLoadInterval('h','i1',15,30,150)])
    r=integrate_station_power('h',10,20,profile=p,capacity_kw=1100,bus_charges=[(10,20)],battery_charges=[(12,30)],bus_charging_power_kw=500,battery_charging_power_kw=6)
    assert [s.total_load_kw for s in r.segments] == [600,606,656]
    assert r.overload_kw_min == 0 and r.peak_load_kw == 656

def test_overload_numeric():
    p=StationBaseLoadProfile([StationBaseLoadInterval('h','i0',0,20,180)])
    r=integrate_station_power('h',0,10,profile=p,capacity_kw=1100,bus_charges=[(0,10),(0,10)],battery_charges=[(0,10)]*6,bus_charging_power_kw=500,battery_charging_power_kw=2)
    assert r.peak_load_kw == 1192
    assert r.overload_kw_min == pytest.approx(920)
    assert r.overload_duration_min == 10
