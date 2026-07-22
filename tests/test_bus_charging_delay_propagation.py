from tests.bus_event_chain_helpers import make_env, run_env

def test_charging_delay_shifts_next_arrival_and_adds_energy(tmp_path):
    env=make_env(tmp_path, stops=6); run_env(env, choose_charge=True)
    charged=[r for r in env.bus_trace.rows if r.charging_duration_min>0]
    assert charged and env.bus_charging_energy_kwh > 0
    r=charged[0]; st=env.runtime_trip_states[r.trip_id]; run,_=env._segment(r.trip_id,r.stop_index,r.stop_index+1)
    assert st.actual_arrival_times[r.stop_index+1] == r.actual_departure + run
