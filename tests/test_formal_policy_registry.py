from evaluation.formal_policy_registry import *
def test_registry_unique_and_required_mappings():
    assert len(FORMAL_METHOD_REGISTRY)==len(set(FORMAL_METHOD_REGISTRY))
    assert get_formal_policy_spec('mappo_env').expected_algorithm=='four_agent_asynchronous_mappo_env'
    assert get_formal_policy_spec('mappo_rlaif_assignment').expected_rlaif_scope=='assignment'
    assert get_formal_policy_spec('mappo_rlaif_all').enabled_reward_agents==('assignment','truck','bus','station')
def test_duplicate_learned_checkpoint_rejected():
    a=get_formal_policy_spec('mappo_env').with_checkpoint('same.pt')
    b=get_formal_policy_spec('mappo_rlaif_assignment').with_checkpoint('same.pt')
    import pytest
    with pytest.raises(PolicyCheckpointValidationError): validate_unique_learned_checkpoints([a,b])
def test_smoke_metadata_rejected():
    import pytest
    with pytest.raises(PolicyCheckpointValidationError): validate_policy_checkpoint(get_formal_policy_spec('mappo_env'), {'run_classification':'smoke','algorithm':'four_agent_asynchronous_mappo_env','rlaif_scope':'none','enabled_reward_agents':[]})
