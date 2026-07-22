from __future__ import annotations
import copy, csv, hashlib, json, math, os, platform, random, subprocess, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch.utils.data import DataLoader
from training.event_schema import EVENT_NAME_TO_ID, REQUIRED_EVENT_COVERAGE, OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION
from rlaif.multi_agent_reward_model import AgentRewardModel, bradley_terry_loss
from rlaif.reward_model_dataset import RewardPairDataset, dataset_hash
from rlaif.reward_model_normalization import FeatureNormalization, apply_feature_normalization
CHECKPOINT_SCHEMA_VERSION=1
STATUSES={'passed','failed_insufficient_data','failed_validation_accuracy','failed_test_accuracy','failed_missing_event_coverage','failed_nonfinite_metrics','smoke_only'}
@dataclass(frozen=True)
class RewardTrainingResult:
    best_epoch:int; last_epoch:int; train_metrics:dict[str,float]; validation_metrics:dict[str,float]; test_metrics:dict[str,float]; checkpoint_path:str; validation_status:str

def set_reward_training_seed(seed:int)->torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True, warn_only=True); g=torch.Generator(); g.manual_seed(seed); return g

def _batch(items):
    return {k: v for k,v in {'state':torch.stack([x.state_features for x in items]), 'a':torch.stack([x.candidate_a_features for x in items]), 'b':torch.stack([x.candidate_b_features for x in items]), 'event':torch.tensor([x.event_type_id for x in items]), 'target':torch.stack([x.target_a_preferred for x in items])}.items()}

def _counts(ds):
    by_event={}; by_source={}; scenarios=set(); episodes=set(); states=set()
    rev={v:k for k,v in EVENT_NAME_TO_ID.items()}
    for e in ds.examples:
        by_event[rev.get(e.event_type_id,str(e.event_type_id))]=by_event.get(rev.get(e.event_type_id,str(e.event_type_id)),0)+1; by_source[e.label_source]=by_source.get(e.label_source,0)+1; scenarios.add(e.scenario_id); episodes.add(e.episode_id); states.add(e.state_id)
    return by_event,by_source,len(scenarios),len(episodes),len(states)

def evaluate_agent_reward_model(model, ds:RewardPairDataset, sn:FeatureNormalization, cn:FeatureNormalization, *, batch_size:int=256, device='cpu', event_filter:int|None=None):
    model.eval(); losses=[]; margins=[]; correct=den=0
    with torch.no_grad():
      for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=_batch):
        mask=torch.ones_like(batch['event'],dtype=torch.bool) if event_filter is None else batch['event'].eq(int(event_filter))
        if not bool(mask.any()): continue
        s=apply_feature_normalization(batch['state'][mask],sn,feature_names=ds.state_feature_names).to(device); a=apply_feature_normalization(batch['a'][mask],cn,feature_names=ds.candidate_feature_names).to(device); b=apply_feature_normalization(batch['b'][mask],cn,feature_names=ds.candidate_feature_names).to(device); ev=batch['event'][mask].to(device); t=batch['target'][mask].to(device)
        sa=model(s,ev,a); sb=model(s,ev,b); loss=bradley_terry_loss(sa,sb,t); losses.append(float(loss.item())*len(t)); diff=(sa-sb).detach().cpu(); pref=torch.where(t.cpu()>0.5,diff,-diff); margins += [float(x) for x in pref]
        pred=torch.sign(diff); truth=torch.where(t.cpu()>0.5, torch.ones_like(diff), -torch.ones_like(diff)); nt=pred.ne(0); correct += int((pred[nt]==truth[nt]).sum()); den += int(nt.sum())
    by_event,by_source,sc,ep,st=_counts(ds)
    pair_count=len(margins); avg=sum(margins)/pair_count if pair_count else float('nan'); med=float(np.median(margins)) if pair_count else float('nan')
    return {'loss':sum(losses)/max(pair_count,1),'pairwise_accuracy':correct/den if den else float('nan'),'average_margin':avg,'median_margin':med,'positive_margin_fraction':sum(m>0 for m in margins)/max(pair_count,1),'pair_count':float(pair_count),'scenario_count':float(sc),'episode_count':float(ep),'state_count':float(st),'counts_by_event':by_event,'counts_by_label_source':by_source}

def determine_validation_status(agent_type, train_metrics, val_metrics, test_metrics, per_event_metrics, cfg):
    if cfg.get('run_classification')=='smoke': return 'smoke_only'
    v=cfg.get('validation',{})
    vals=[train_metrics.get('loss'),val_metrics.get('loss'),test_metrics.get('loss'),val_metrics.get('pairwise_accuracy'),test_metrics.get('pairwise_accuracy')]
    if any(not math.isfinite(float(x)) for x in vals if x is not None): return 'failed_nonfinite_metrics'
    if train_metrics['pair_count'] < v.get('minimum_training_pairs',0) or val_metrics['pair_count'] < v.get('minimum_validation_pairs',0) or test_metrics['pair_count'] < v.get('minimum_test_pairs',0): return 'failed_insufficient_data'
    req=REQUIRED_EVENT_COVERAGE.get(agent_type,set())
    if v.get('require_all_event_types_present',False) and (req-set(val_metrics['counts_by_event']) or req-set(test_metrics['counts_by_event'])): return 'failed_missing_event_coverage'
    if val_metrics['pairwise_accuracy'] < v.get('minimum_validation_pairwise_accuracy',0): return 'failed_validation_accuracy'
    if test_metrics['pairwise_accuracy'] < v.get('minimum_test_pairwise_accuracy',0): return 'failed_test_accuracy'
    if v.get('require_positive_average_margin',False) and (val_metrics['average_margin']<=0 or test_metrics['average_margin']<=0): return 'failed_validation_accuracy'
    for ev,m in per_event_metrics.items():
        if m['pairwise_accuracy'] < v.get('minimum_event_pairwise_accuracy',0): return 'failed_validation_accuracy'
    return 'passed'

def _sha_file(p):
    p=Path(p) if p else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None

def _git_sha():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return 'unknown'

def train_agent_reward_model(*, agent_type:str, train_dataset:RewardPairDataset, validation_dataset:RewardPairDataset, test_dataset:RewardPairDataset, state_normalization:FeatureNormalization, candidate_normalization:FeatureNormalization, config:dict, output_path:str)->RewardTrainingResult:
    tr=config.get('training',{}); seed=int(tr.get('seed',1)); gen=set_reward_training_seed(seed); device=torch.device('cpu' if tr.get('device','auto')=='cpu' or not torch.cuda.is_available() else 'cuda')
    model_cfg=config.get('model',{}); model=AgentRewardModel(state_dim=train_dataset.state_dim,candidate_dim=train_dataset.candidate_dim,num_event_types=len(EVENT_NAME_TO_ID),event_embedding_dim=int(model_cfg.get('event_embedding_dim',16)),hidden_dims=tuple(model_cfg.get('hidden_dims',[64,64])),dropout=float(model_cfg.get('dropout',0.0))).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(tr.get('learning_rate',1e-3)),weight_decay=float(tr.get('weight_decay',0.0)))
    best=None; best_loss=float('inf'); best_epoch=0; patience=0; hist=[]; epochs=int(tr.get('epochs',10)); batch_size=int(tr.get('batch_size',32)); minimp=float(tr.get('minimum_improvement',1e-4)); maxgn=float(tr.get('max_grad_norm',1.0))
    for epoch in range(1,epochs+1):
        model.train(); grads=[]
        for batch in DataLoader(train_dataset,batch_size=batch_size,shuffle=True,generator=gen,collate_fn=_batch):
            s=apply_feature_normalization(batch['state'],state_normalization,feature_names=train_dataset.state_feature_names).to(device); a=apply_feature_normalization(batch['a'],candidate_normalization,feature_names=train_dataset.candidate_feature_names).to(device); b=apply_feature_normalization(batch['b'],candidate_normalization,feature_names=train_dataset.candidate_feature_names).to(device); ev=batch['event'].to(device); t=batch['target'].to(device)
            opt.zero_grad(); loss=bradley_terry_loss(model(s,ev,a),model(s,ev,b),t); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),maxgn); opt.step(); grads.append(float(gn))
        tm=evaluate_agent_reward_model(model,train_dataset,state_normalization,candidate_normalization,batch_size=batch_size,device=device); vm=evaluate_agent_reward_model(model,validation_dataset,state_normalization,candidate_normalization,batch_size=batch_size,device=device)
        improved=vm['loss'] < best_loss-minimp
        if improved: best_loss=vm['loss']; best=copy.deepcopy(model.state_dict()); best_epoch=epoch; patience=0
        else: patience+=1
        hist.append({'epoch':epoch,'training_loss':tm['loss'],'training_pairwise_accuracy':tm['pairwise_accuracy'],'training_average_margin':tm['average_margin'],'validation_loss':vm['loss'],'validation_pairwise_accuracy':vm['pairwise_accuracy'],'validation_average_margin':vm['average_margin'],'learning_rate':opt.param_groups[0]['lr'],'gradient_norm':sum(grads)/max(len(grads),1),'best_so_far':improved,'patience_counter':patience})
        if patience>=int(tr.get('early_stopping_patience',epochs+1)): break
    if best is not None: model.load_state_dict(best)
    train_m=evaluate_agent_reward_model(model,train_dataset,state_normalization,candidate_normalization,batch_size=batch_size,device=device); val_m=evaluate_agent_reward_model(model,validation_dataset,state_normalization,candidate_normalization,batch_size=batch_size,device=device); test_m=evaluate_agent_reward_model(model,test_dataset,state_normalization,candidate_normalization,batch_size=batch_size,device=device)
    per={}
    for ev in REQUIRED_EVENT_COVERAGE.get(agent_type,set()): per[ev]=evaluate_agent_reward_model(model,test_dataset,state_normalization,candidate_normalization,batch_size=batch_size,device=device,event_filter=EVENT_NAME_TO_ID[ev])
    status=determine_validation_status(agent_type,train_m,val_m,test_m,per,config)
    scores=[]
    with torch.no_grad():
      for e in train_dataset.examples:
        s=apply_feature_normalization(e.state_features[None],state_normalization,feature_names=train_dataset.state_feature_names).to(device); a=apply_feature_normalization(e.candidate_a_features[None],candidate_normalization,feature_names=train_dataset.candidate_feature_names).to(device); scores.append(float(model(s,torch.tensor([e.event_type_id],device=device),a).item()))
    mean=float(np.mean(scores)) if scores else 0.0; std=float(np.std(scores)) if scores and np.std(scores)>1e-8 else 1.0
    ck={'checkpoint_type':'agent_reward_model','checkpoint_schema_version':CHECKPOINT_SCHEMA_VERSION,'run_classification':config.get('run_classification','formal'),'validation_status':status,'agent_type':agent_type,'compatible_event_types':sorted(REQUIRED_EVENT_COVERAGE[agent_type]),'model_class':'AgentRewardModel','model_architecture':model_cfg,'model_state_dict':model.state_dict(),'state_feature_names':list(train_dataset.state_feature_names),'candidate_feature_names':list(train_dataset.candidate_feature_names),'state_feature_dim':train_dataset.state_dim,'candidate_feature_dim':train_dataset.candidate_dim,'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION,'event_name_to_id':dict(EVENT_NAME_TO_ID),'state_normalization_mean':list(state_normalization.mean),'state_normalization_std':list(state_normalization.std),'candidate_normalization_mean':list(candidate_normalization.mean),'candidate_normalization_std':list(candidate_normalization.std),'reward_output_training_mean':mean,'reward_output_training_std':std,'training_config':tr,'split_config':config.get('split',{}),'validation_config':config.get('validation',{}),'preference_file_hash':config.get('preference_file_hash'),'training_data_hash':dataset_hash(train_dataset),'split_manifest_hash':config.get('split_manifest_hash'),'best_epoch':best_epoch,'last_epoch':epoch,'best_validation_loss':best_loss,'train_metrics':train_m,'validation_metrics':val_m,'test_metrics':test_m,'per_event_metrics':per,'excluded_label_counts':dict(train_dataset.report.excluded_outcomes),'label_source_counts':train_m['counts_by_label_source'],'training_seed':seed,'actual_device':str(device),'cuda_available':torch.cuda.is_available(),'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'code_commit_sha':_git_sha(),'PyTorch_version':torch.__version__,'creation_timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True); torch.save(ck,out)
    if hist:
      with (out.parent/'training_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    (out.parent/'metrics.json').write_text(json.dumps({'train':train_m,'validation':val_m,'test':test_m,'per_event':per,'validation_status':status},indent=2))
    return RewardTrainingResult(best_epoch,epoch,train_m,val_m,test_m,str(out),status)
