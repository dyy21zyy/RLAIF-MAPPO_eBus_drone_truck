from envs.dynamics.passenger_dynamics import generate_arrival_events

def test_destinations_are_downstream():
    stops=[f"s{i}" for i in range(6)]
    _,events,_=generate_arrival_events(stops,60,8, rates={s:0.6 for s in stops})
    pos={s:i for i,s in enumerate(stops)}
    assert events
    assert all(pos[e.destination_stop_id] > pos[e.origin_stop_id] for e in events)
