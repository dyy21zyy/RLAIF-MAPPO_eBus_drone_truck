from tests.test_station_dispatch_matching import env
from envs.action_generators.station_actions import generate_station_operation_candidates, projected_load

def test_power_overload_allowed_but_exposed():
    e=env(); st=e.stations['s']; st.power_capacity_kw=1; st.active_bus_charges=[100]
    c=generate_station_operation_candidates(e,'s')
    assert any(x.feasible and x.projected_overload > 0 for x in c)
    assert projected_load(e,st) > st.power_capacity_kw
