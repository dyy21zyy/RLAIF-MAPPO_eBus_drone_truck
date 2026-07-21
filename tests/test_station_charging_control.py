from tests.test_station_dispatch_matching import env
from envs.dynamics.station_dynamics import start_battery_charging
from envs.delivery_env import RuntimeBatteryState
import pytest

def test_station_action_starts_charging_and_slot_45_min_nonpreemptive():
    e=env(); st=e.stations['s']; b=st.battery_states[0]; b.status='DEPLETED'
    start_battery_charging(e,st,b,10.0)
    assert b.status=='CHARGING'; assert b.charge_completion_time_min==55.0
    with pytest.raises(ValueError): start_battery_charging(e,st,b,20.0)

def test_seventh_simultaneous_charge_is_infeasible():
    e=env(); st=e.stations['s']; st.battery_states=[RuntimeBatteryState(f'b{i}','s','DEPLETED') for i in range(7)]
    for b in st.battery_states[:6]: start_battery_charging(e,st,b,0.0)
    with pytest.raises(ValueError): start_battery_charging(e,st,st.battery_states[6],0.0)
