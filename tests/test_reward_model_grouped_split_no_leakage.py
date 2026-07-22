import pytest
from rlaif.grouped_split import grouped_split

def rows(): return [{'scenario_id':f's{i//4}','episode_id':f'e{i//2}','state_id':f'st{i%2}','preference_id':str(i)} for i in range(12)]
def test_same_state_never_crosses_splits():
    sp=grouped_split(rows(),.5,.25,.25,1,'state'); seen={};
    for split, rs in sp['records'].items():
        for r in rs: seen.setdefault((r['scenario_id'],r['episode_id'],r['state_id']),split)==split
    assert set(map(tuple,sp['splits']['train'])).isdisjoint(set(map(tuple,sp['splits']['validation'])))
def test_same_episode_never_crosses_splits_in_episode_mode():
    sp=grouped_split(rows(),.5,.25,.25,1,'episode'); assert all(len(g)==2 for gs in sp['splits'].values() for g in gs)
def test_same_scenario_never_crosses_splits_in_scenario_mode():
    sp=grouped_split(rows(),.5,.25,.25,1,'scenario'); assert all(len(g)==1 for gs in sp['splits'].values() for g in gs)
def test_same_seed_reproduces_split(): assert grouped_split(rows(),.5,.25,.25,7,'scenario')['splits']==grouped_split(rows(),.5,.25,.25,7,'scenario')['splits']
def test_invalid_split_fractions_fail():
    with pytest.raises(ValueError): grouped_split(rows(),.8,.3,.1,1,'state')
