import pytest
from evaluation.formal_policy_registry import get_formal_policy_spec, validate_policy_checkpoint, PolicyCheckpointValidationError

def meta(**kw):
    d={'checkpoint_schema_version':1,'algorithm':'four_agent_asynchronous_mappo_env','rlaif_scope':'none','enabled_reward_agents':[],'training_seed':1,'training_scenario_bank_hash':'b','resolved_training_config_hash':'c','code_commit':'d','run_classification':'formal'}; d.update(kw); return d

def test_wrong_algorithm_fails():
    with pytest.raises(PolicyCheckpointValidationError): validate_policy_checkpoint(get_formal_policy_spec('mappo_env').with_checkpoint('x',training_seed=1), meta(algorithm='bad'))
def test_wrong_scope_fails():
    with pytest.raises(PolicyCheckpointValidationError): validate_policy_checkpoint(get_formal_policy_spec('mappo_env').with_checkpoint('x',training_seed=1), meta(rlaif_scope='all'))
def test_wrong_seed_fails():
    with pytest.raises(PolicyCheckpointValidationError): validate_policy_checkpoint(get_formal_policy_spec('mappo_env').with_checkpoint('x',training_seed=2), meta())
def test_diagnostic_checkpoint_fails_formal_validation():
    with pytest.raises(PolicyCheckpointValidationError): validate_policy_checkpoint(get_formal_policy_spec('mappo_env').with_checkpoint('x',training_seed=1), meta(run_classification='diagnostic'))
