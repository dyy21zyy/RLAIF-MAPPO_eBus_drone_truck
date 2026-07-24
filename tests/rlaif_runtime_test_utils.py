from pathlib import Path
import torch
from rlaif.multi_agent_reward_model import AgentRewardModel
from training.event_schema import EVENT_NAME_TO_ID, REQUIRED_EVENT_COVERAGE, OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION

STATE=["s0","s1"]
CAND=["c0","c1"]

def make_ckpt(path: Path, agent='assignment', classification='smoke', validation='smoke_only', state_mean=None, cand_mean=None, reward_mean=0.0, reward_std=1.0, state_names=None, cand_names=None):
    state_names = list(state_names or STATE)
    cand_names = list(cand_names or CAND)
    m=AgentRewardModel(state_dim=len(state_names),candidate_dim=len(cand_names),num_event_types=len(EVENT_NAME_TO_ID),event_embedding_dim=4,hidden_dims=(4,),dropout=0.0)
    for p in m.parameters(): torch.nn.init.constant_(p, 0.1)
    ck={
      'checkpoint_type':'agent_reward_model','checkpoint_schema_version':1,'run_classification':classification,'validation_status':validation,'agent_type':agent,'compatible_event_types':sorted(REQUIRED_EVENT_COVERAGE[agent]),'model_architecture':{'event_embedding_dim':4,'hidden_dims':[4],'dropout':0.0},'model_state_dict':m.state_dict(),
      'state_feature_names':state_names,'candidate_feature_names':cand_names,'state_feature_dim':len(state_names),'candidate_feature_dim':len(cand_names),'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION,'event_name_to_id':dict(EVENT_NAME_TO_ID),
      'state_normalization_mean': state_mean or [0.0]*len(state_names), 'state_normalization_std':[1.0]*len(state_names), 'candidate_normalization_mean': cand_mean or [0.0]*len(cand_names), 'candidate_normalization_std':[1.0]*len(cand_names), 'reward_output_training_mean':reward_mean, 'reward_output_training_std':reward_std,
      'preference_file_hash':'pref','training_data_hash':'data','split_manifest_hash':'split','train_metrics':{},'validation_metrics':{},'test_metrics':{},'training_seed':1,'code_commit_sha':'test'}
    torch.save(ck,path); return path

def cfg(tmp_path, scope='assignment', classification='smoke', validation='smoke_only'):
    agents={}
    selected=['assignment'] if scope=='assignment' else ['assignment','truck','bus','station']
    for a in ['assignment','truck','bus','station']:
        enabled=a in selected
        ck=make_ckpt(tmp_path/f'{a}.pt', a, classification, validation) if enabled else None
        agents[a]={'enabled':enabled,'checkpoint':str(ck) if ck else None,'lambda':1.5,'reward_clip':0.5}
    return {'run_classification':classification,'rlaif':{'enabled':True,'scope':scope,'fallback_to_env_reward':False,'fail_on_invalid_reward_model':True,'agents':agents}}
