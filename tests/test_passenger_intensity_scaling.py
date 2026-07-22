from envs.dynamics.passenger_demand import PassengerTemporalBlock, effective_passenger_rate, generate_time_dependent_arrivals

def test_effective_rate_can_exceed_baseline_upper_bound():
    assert effective_passenger_rate(0.60,2.0,1.0) == 1.20

def test_intensity_doubles_expected_demand_with_same_baselines():
    blocks=(PassengerTemporalBlock('all',0,120,1.0),); stops=['a','b']; rates={'a':0.3,'b':0.3}
    low=[]; high=[]
    for seed in range(100,150):
        low.append(len(generate_time_dependent_arrivals(stops,horizon_min=120,baseline_rates=rates,demand_intensity=1,temporal_blocks=blocks,seed=seed)))
        high.append(len(generate_time_dependent_arrivals(stops,horizon_min=120,baseline_rates=rates,demand_intensity=2,temporal_blocks=blocks,seed=seed)))
    ratio=sum(high)/sum(low)
    assert 1.75 < ratio < 2.25
