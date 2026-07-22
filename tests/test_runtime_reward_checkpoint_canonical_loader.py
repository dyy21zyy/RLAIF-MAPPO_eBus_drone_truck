import json, pytest, torch
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from training.reward_model_wrapper import RewardCheckpointError, RewardCheckpointCompatibilityError, RewardCheckpointValidationError
from tests.rlaif_runtime_test_utils import make_ckpt

def test_canonical_all_agents_load(tmp_path):
    for agent in ['assignment','truck','bus','station']:
        p=make_ckpt(tmp_path/f'{agent}.pt', agent)
        assert RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type=agent, formal_mode=False).agent_type == agent

def test_legacy_json_wrong_type_and_smoke_formal_rejected(tmp_path):
    j=tmp_path/'legacy.json'; j.write_text(json.dumps({'smoke_placeholder':True}))
    with pytest.raises(RewardCheckpointValidationError): RuntimeAgentRewardModel.from_checkpoint(j, expected_agent_type='assignment', formal_mode=False)
    p=make_ckpt(tmp_path/'bad.pt'); ck=torch.load(p,weights_only=False); ck['checkpoint_type']='other'; torch.save(ck,p)
    with pytest.raises(RewardCheckpointCompatibilityError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=False)
    p=make_ckpt(tmp_path/'smoke.pt')
    with pytest.raises(RewardCheckpointValidationError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=True)
    p=make_ckpt(tmp_path/'failed.pt', classification='formal', validation='failed')
    with pytest.raises(RewardCheckpointValidationError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=True)
