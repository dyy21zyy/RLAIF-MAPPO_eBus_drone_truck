import json, torch, pytest
from pathlib import Path
from training.reward_model_wrapper import load_strict_agent_reward_checkpoint, RewardCheckpointValidationError, RewardCheckpointCompatibilityError
def test_smoke_checkpoint_contract_if_present():
 p=Path('results/smoke/reward_models/reward_assignment.pt')
 if not p.exists(): pytest.skip('smoke checkpoint not generated')
 ck,model=load_strict_agent_reward_checkpoint(p,agent_type='assignment',formal=False)
 assert ck['model_state_dict'] and ck['state_normalization_mean'] and ck['reward_output_training_std']>0
 with pytest.raises(RewardCheckpointValidationError): load_strict_agent_reward_checkpoint(p,agent_type='assignment',formal=True)
def test_placeholder_json_rejected(tmp_path):
 p=tmp_path/'placeholder.json'; p.write_text(json.dumps({'smoke_placeholder':True}))
 with pytest.raises(RewardCheckpointValidationError, match='legacy placeholder'): load_strict_agent_reward_checkpoint(p,agent_type='assignment',formal=False)
