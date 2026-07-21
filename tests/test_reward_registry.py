import pytest, torch
from pathlib import Path
from rlaif.multi_agent_reward_model import MultiAgentRewardModel
from rlaif.reward_registry import RewardRegistry
from training.reward_model_wrapper import RewardModelCheckpointError

def ckpt(path,agent='assignment',events=['PARCEL_RELEASE'],mean=10.0,std=2.0):
    m=MultiAgentRewardModel(2,2,events)
    torch.save({'agent_type':agent,'compatible_event_types':events,'state_schema_version':'v2','candidate_schema_version':'v2','state_feature_dim':2,'candidate_feature_dim':2,'state_feature_names':['s1','s2'],'candidate_feature_names':['c1','c2'],'state_feature_mean':torch.zeros(2),'state_feature_std':torch.ones(2),'candidate_feature_mean':torch.zeros(2),'candidate_feature_std':torch.ones(2),'reward_mean':mean,'reward_std':std,'training_data_hash':'x','split_manifest_hash':'y','validation_metrics':{},'model_config':{'hidden_dims':[64,64]},'model_state_dict':m.state_dict()}, path)

def test_wrong_agent_event_schema_and_missing_fail_closed(tmp_path):
    p=tmp_path/'truck.pt'; ckpt(p,agent='truck',events=['TRUCK_AVAILABLE'])
    with pytest.raises(RewardModelCheckpointError): RewardRegistry({'rlaif':{'enabled':True,'fail_on_invalid_reward_model':True,'agents':{'assignment':{'checkpoint':str(p),'lambda':.2}}}})
    with pytest.raises(RewardModelCheckpointError): RewardRegistry({'rlaif':{'enabled':True,'fail_on_invalid_reward_model':True,'agents':{'truck':{'checkpoint':str(tmp_path/'missing.pt'),'lambda':.2}}}})
    reg=RewardRegistry({'rlaif':{'enabled':True,'fail_on_invalid_reward_model':True,'agents':{'truck':{'checkpoint':str(p),'lambda':.2}}}})
    with pytest.raises(RewardModelCheckpointError): reg.score('truck','WRONG',[0,0],[0,0])

def test_normalization_before_clipping_and_inactive_no_reward(tmp_path):
    p=tmp_path/'a.pt'; ckpt(p)
    reg=RewardRegistry({'rlaif':{'enabled':True,'fail_on_invalid_reward_model':True,'agents':{'assignment':{'checkpoint':str(p),'lambda':.5,'reward_clip':1.0},'truck':{'enabled':False,'checkpoint':str(p),'lambda':.5}}}})
    total,learned=reg.total_reward('assignment','PARCEL_RELEASE',3.0,[0,0],[0,0])
    assert -1.0 <= learned <= 1.0 and total == 3.0 + .5*learned
    assert reg.total_reward('truck','TRUCK_AVAILABLE',4.0,[0,0],[0,0]) == (4.0,0.0)
