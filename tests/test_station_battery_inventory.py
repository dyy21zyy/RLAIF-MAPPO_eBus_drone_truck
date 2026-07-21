from tests.test_station_dispatch_matching import env
from envs.dynamics.station_dynamics import dispatch_drone

def test_dispatch_consumes_full_and_return_depletes_without_charging():
    e=env(); st=e.stations['s']; d=st.drone_states[0]; b=st.battery_states[0]; p=e.parcels['p0']
    dispatch_drone(e,st,d,p,b,0.0)
    assert b.status=='IN_USE'; assert d.status=='IN_MISSION'; assert not any(k=='battery_ready' for _,k,_ in e.events)
    b.status='DEPLETED'; d.status='AVAILABLE'
    assert sum(x.status in {'FULL','IN_USE','DEPLETED','CHARGING'} for x in st.battery_states)==3
