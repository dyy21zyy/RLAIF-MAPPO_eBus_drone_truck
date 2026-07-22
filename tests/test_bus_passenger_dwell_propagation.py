from tests.bus_event_chain_helpers import make_env, run_env

def test_ordinary_passenger_dwell_shifts_downstream(tmp_path):
    env=make_env(tmp_path, stops=6); run_env(env)
    rows=[r for r in env.bus_trace.rows if r.trip_id=='test_trip_000']
    ordinary=[r for r in rows if not r.integrated_station and r.passenger_boarding]
    assert ordinary
    r=ordinary[0]; st=env.runtime_trip_states[r.trip_id]; run,_=env._segment(r.trip_id,r.stop_index,r.stop_index+1)
    assert st.actual_arrival_times[r.stop_index+1] == r.actual_departure + run
