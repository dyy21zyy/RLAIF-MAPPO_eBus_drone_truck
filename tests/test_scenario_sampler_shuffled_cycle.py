from training.scenario_sampler import ScenarioSampler

def test_shuffled_cycle_visits_every_scenario_once_per_cycle():
    ids=['a','b','c','d']; s=ScenarioSampler(ids,mode='shuffled_cycle',seed=3)
    first=[s.next_scenario_id() for _ in ids]; second=[s.next_scenario_id() for _ in ids]
    assert set(first)==set(ids) and set(second)==set(ids)
    assert first != second

def test_uniform_random_uses_declared_scenarios():
    ids=['x','y']; s=ScenarioSampler(ids,mode='uniform_random',seed=9)
    assert set(s.next_scenario_id() for _ in range(20)) <= set(ids)
