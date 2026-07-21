from training.mappo_async import RLAIF_AGENT_TYPES

def test_rlaif_agent_scope_all_four():
    assert RLAIF_AGENT_TYPES == {'assignment','truck','bus','station'}
