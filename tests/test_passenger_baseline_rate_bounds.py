from envs.dynamics.passenger_demand import sample_truncated_normal_rates

def test_baseline_rates_in_bounds_and_reproducible():
    stops=[f's{i}' for i in range(200)]
    a=sample_truncated_normal_rates(stops,seed=1); b=sample_truncated_normal_rates(stops,seed=1)
    assert a==b
    assert min(a.values()) >= 0.05 and max(a.values()) <= 0.60
