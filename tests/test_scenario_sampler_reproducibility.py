from training.scenario_sampler import ScenarioSampler

def test_same_seed_reproduces_shuffled_cycle_order():
    ids=['a','b','c']; a=ScenarioSampler(ids,mode='shuffled_cycle',seed=7); b=ScenarioSampler(ids,mode='shuffled_cycle',seed=7)
    assert [a.next_scenario_id() for _ in range(8)] == [b.next_scenario_id() for _ in range(8)]

def test_sequential_order_is_stable():
    s=ScenarioSampler(['a','b'],mode='sequential',seed=1)
    assert [s.next_scenario_id() for _ in range(5)] == ['a','b','a','b','a']
