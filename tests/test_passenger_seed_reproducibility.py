from envs.dynamics.passenger_demand import PassengerTemporalBlock, generate_time_dependent_arrivals

def test_same_seed_reproducible_different_seed_changes():
    blocks=(PassengerTemporalBlock('all',0,20,1),); rates={'a':1,'b':1}
    a=generate_time_dependent_arrivals(['a','b'],horizon_min=20,baseline_rates=rates,demand_intensity=1,temporal_blocks=blocks,seed=2)
    b=generate_time_dependent_arrivals(['a','b'],horizon_min=20,baseline_rates=rates,demand_intensity=1,temporal_blocks=blocks,seed=2)
    c=generate_time_dependent_arrivals(['a','b'],horizon_min=20,baseline_rates=rates,demand_intensity=1,temporal_blocks=blocks,seed=3)
    assert a==b and a!=c
