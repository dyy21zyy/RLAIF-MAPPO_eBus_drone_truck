import pytest
from envs.dynamics.passenger_demand import PassengerTemporalBlock, validate_temporal_profile, generate_time_dependent_arrivals

def blocks(): return (PassengerTemporalBlock('low',0,60,0.1), PassengerTemporalBlock('high',60,120,3.0))

def test_temporal_profile_validation():
    assert validate_temporal_profile(blocks(), horizon_min=120)
    with pytest.raises(ValueError): validate_temporal_profile((PassengerTemporalBlock('a',0,70,1),PassengerTemporalBlock('b',60,120,1)), horizon_min=120)
    with pytest.raises(ValueError): validate_temporal_profile((PassengerTemporalBlock('a',0,50,1),PassengerTemporalBlock('b',60,120,1)), horizon_min=120)
    with pytest.raises(ValueError): validate_temporal_profile((PassengerTemporalBlock('a',0,120,0),), horizon_min=120)

def test_arrivals_are_blocked_downstream_and_final_stop_empty():
    stop_ids=['s0','s1','s2']; rates={s:0.5 for s in stop_ids}
    ev=generate_time_dependent_arrivals(stop_ids,horizon_min=120,baseline_rates=rates,demand_intensity=1.0,temporal_blocks=blocks(),seed=7)
    assert ev
    idx={s:i for i,s in enumerate(stop_ids)}
    assert all(idx[e.destination_stop_id] > idx[e.origin_stop_id] for e in ev)
    assert all(e.origin_stop_id != 's2' for e in ev)
    by={b:sum(e.passenger_count for e in ev if e.block_id==b) for b in ['low','high']}
    assert by['high'] > by['low']
    for e in ev:
        if e.block_id=='low': assert 0 <= e.arrival_time_min <= 60
        if e.block_id=='high': assert 60 <= e.arrival_time_min <= 120
