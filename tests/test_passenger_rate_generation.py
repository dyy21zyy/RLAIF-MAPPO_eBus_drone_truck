from envs.dynamics.passenger_dynamics import generate_arrival_events

def test_stop_rates_within_bounds_and_seed_reproducible():
    stops=[f"s{i}" for i in range(8)]
    rates,a,prov=generate_arrival_events(stops,120,123)
    assert all(0.05 <= r <= 0.60 for r in rates.values())
    rates2,b,_=generate_arrival_events(stops,120,123)
    assert rates == rates2
    assert [e.__dict__ for e in a] == [e.__dict__ for e in b]
    _,c,_=generate_arrival_events(stops,120,124)
    assert [e.__dict__ for e in a] != [e.__dict__ for e in c]
    assert prov["seed"] == 123
