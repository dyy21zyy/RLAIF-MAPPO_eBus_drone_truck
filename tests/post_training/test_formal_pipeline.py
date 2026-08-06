import json
from pathlib import Path
import pytest
from experiments.run_hard_locker_post_training_experiment import PHASES,build_plan
from experiments.post_training import METHODS
from experiments.post_training.config_resolver import training_jobs
from experiments.post_training.benchmark import build_rows
from experiments.post_training.preference_collection import make_record
from experiments.post_training.readiness import validate_seed_pair,ReadinessError
from experiments.post_training.analysis import COMPARISONS

def test_phase_order_and_nine_jobs():
 assert [x['phase'] for x in build_plan()]==list(range(12)); assert len(PHASES)==12
 jobs=training_jobs(); assert len(jobs)==9
 assert {m:sum(j['method_id']==m for j in jobs) for m in METHODS}=={m:3 for m in METHODS}

def test_preference_base_lineage_and_feasible(tmp_path):
 p=tmp_path/'p.pt';p.write_bytes(b'x')
 r=make_record(scenario_id='s',scenario_content_hash='h',seed=1,checkpoint=p,state_features=[0],candidate_a=0,candidate_b=1,candidate_feature_names=['x'],action_mask=[True,True],event_type='TD',commit='c',split='train')
 assert r['agent_type']=='assignment' and r['source_policy_method_id']=='mappo_env_post_base' and r['source_policy_checkpoint_sha256']

def _configs():
 common={'seed':1,'total_episodes':2,'rollout_episodes':1,'lr_actor':1,'lr_critic':1,'gamma':.9,'gae_lambda':.9,'clip_eps':.2,'ppo_epochs':1,'batch_size':2,'entropy_coef':.1,'value_coef':.5,'max_grad_norm':1,'event_time_reference_min':5,'device_policy':'cpu','scenario_bank':'b','reward_scale_artifact':'r','network_architecture':[1]}
 init={'load_actors':True,'load_critic':True,'load_optimizers':False}
 b={'training':dict(common),'checkpoint_sha256':'abc'}
 c={'training':{**common,'initialization':dict(init)},'parent_checkpoint_sha256':'abc','rlaif':{'enabled':False}}
 r={'training':{**common,'initialization':dict(init)},'parent_checkpoint_sha256':'abc','rlaif':{'enabled':True,'scope':'assignment','validation_status':'validated','agents':{'assignment':{'enabled':True},'truck':{'enabled':False},'bus':{'enabled':False},'station':{'enabled':False}}}}
 return b,c,r

def test_fair_gate_and_failures():
 b,c,r=_configs();validate_seed_pair(b,c,r)
 r['training']['seed']=2
 with pytest.raises(ReadinessError):validate_seed_pair(b,c,r)

def test_benchmark_cartesian_product():
 policies={}
 for m in METHODS:
  for s in (1,2,3): policies[m,s]={'algorithm':'a','training_stage':'x','checkpoint_path':'p','checkpoint_sha256':'h','reward_scale_artifact_path':'r','reward_scale_artifact_hash':'rh','code_commit':'c'}
 scenarios=[{'scenario_id':str(i),'scenario_content_hash':str(i)} for i in range(100)]
 rows=build_rows(policies,scenarios);assert len(rows)==900;assert len({(x['method_id'],x['training_seed'],x['scenario_id']) for x in rows})==900

def test_preregistered_confirmatory_comparison():
 assert len(COMPARISONS)==3
 assert COMPARISONS[0]=={'comparison_id':'rlaif_post_vs_env_continued','treatment':'mappo_rlaif_assignment_post','baseline':'mappo_env_post_continued','inference':'confirmatory'}

@pytest.mark.parametrize("field", ["total_episodes","rollout_episodes","lr_actor","lr_critic","gamma","gae_lambda","clip_eps","ppo_epochs","batch_size","entropy_coef","value_coef","max_grad_norm","network_architecture","scenario_bank","reward_scale_artifact"])
def test_every_budget_and_hyperparameter_mismatch_fails_closed(field):
 b,c,r=_configs(); r["training"][field] = "tampered"
 with pytest.raises(ReadinessError,match=field): validate_seed_pair(b,c,r)

@pytest.mark.parametrize("field,value", [("load_critic",False),("load_optimizers",True)])
def test_initialization_tampering_fails_closed(field,value):
 b,c,r=_configs(); r["training"]["initialization"][field]=value
 with pytest.raises(ReadinessError): validate_seed_pair(b,c,r)

def test_parent_hash_and_parent_seed_tampering_fail_closed():
 b,c,r=_configs(); c["parent_checkpoint_sha256"]="tampered"
 with pytest.raises(ReadinessError,match="parent hashes"): validate_seed_pair(b,c,r)
 b,c,r=_configs(); r["parent_training_seed"]=2
 with pytest.raises(ReadinessError,match="parent seed"): validate_seed_pair(b,c,r)
