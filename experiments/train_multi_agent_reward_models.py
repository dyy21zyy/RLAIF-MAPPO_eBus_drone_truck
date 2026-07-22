from __future__ import annotations
import argparse,json,hashlib,sys,yaml
from pathlib import Path
import torch
from rlaif.reward_model_dataset import build_reward_pair_dataset
from rlaif.grouped_split import grouped_split
from rlaif.reward_model_normalization import fit_feature_normalization
from rlaif.reward_model_trainer import train_agent_reward_model, determine_validation_status
from training.event_schema import REQUIRED_EVENT_COVERAGE

def sha(p):
    p=Path(p) if p else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None

def load_jsonl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def read_config(p): return yaml.safe_load(Path(p).read_text()) if p else {}
def validate_config(cfg,agent):
    for section in ('run_classification','model','training','validation'): assert section in cfg, f'missing {section}'
    assert agent in REQUIRED_EVENT_COVERAGE
    return True
def split_dataset(ds,cfg):
    sp=cfg.get('split',{}); split=grouped_split(ds.examples,sp.get('train_fraction',.7),sp.get('validation_fraction',.15),sp.get('test_fraction',.15),sp.get('seed',1),sp.get('group_by','scenario'))
    def sub(name): return type(ds)(split['records'][name],agent_type=ds.agent_type,state_feature_names=ds.state_feature_names,candidate_feature_names=ds.candidate_feature_names,report=ds.report)
    return sub('train'),sub('validation'),sub('test'),split

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--preferences'); ap.add_argument('--config',required=True); ap.add_argument('--agent',required=True); ap.add_argument('--output'); ap.add_argument('--split-manifest'); ap.add_argument('--validate-only',action='store_true'); ap.add_argument('--config-only',action='store_true'); ap.add_argument('--device')
    ns=ap.parse_args(argv); cfg=read_config(ns.config); validate_config(cfg,ns.agent)
    if ns.device: cfg.setdefault('training',{})['device']=ns.device
    if ns.config_only:
        print(json.dumps({'config_status':'valid','agent':ns.agent,'run_classification':cfg.get('run_classification')})); return 0
    if not ns.preferences or not Path(ns.preferences).exists():
        status='BLOCKED_MISSING_FINAL_PREFERENCE_DATA' if cfg.get('run_classification')=='formal' else 'missing_preferences'
        print(json.dumps({'validation_status':status})); return 2 if ns.validate_only else 1
    rows=load_jsonl(ns.preferences); ds=build_reward_pair_dataset(rows,agent_type=ns.agent,formal_mode=cfg.get('run_classification')=='formal',require_bus_event_coverage=(ns.agent=='bus' and cfg.get('run_classification')=='formal'))
    train,val,test,split=split_dataset(ds,cfg); cfg['preference_file_hash']=sha(ns.preferences); cfg['split_manifest_hash']=hashlib.sha256(json.dumps(split['splits'],sort_keys=True).encode()).hexdigest()
    sn=fit_feature_normalization(torch.stack([e.state_features for e in train.examples]),feature_names=train.state_feature_names); cn=fit_feature_normalization(torch.cat([torch.stack([e.candidate_a_features for e in train.examples]),torch.stack([e.candidate_b_features for e in train.examples])]),feature_names=train.candidate_feature_names)
    if ns.validate_only:
        zero={'pair_count':float(len(train)),'pairwise_accuracy':0,'loss':0,'average_margin':0,'counts_by_event':{}}
        print(json.dumps({'validation_status':'validated_inputs','split_counts':{'train':len(train),'validation':len(val),'test':len(test)}})); return 0
    out=ns.output or f'results/{cfg.get("run_classification","formal")}/reward_models/reward_{ns.agent}.pt'
    res=train_agent_reward_model(agent_type=ns.agent,train_dataset=train,validation_dataset=val,test_dataset=test,state_normalization=sn,candidate_normalization=cn,config=cfg,output_path=out)
    Path(out).with_name('resolved_config.yaml').write_text(yaml.safe_dump(cfg)); Path(out).with_name('reward_model_run_manifest.json').write_text(json.dumps({'agent':ns.agent,'checkpoint_path':out,'validation_status':res.validation_status,'best_epoch':res.best_epoch},indent=2))
    print(json.dumps(res.__dict__,default=str)); return 0 if res.validation_status in ('passed','smoke_only') else 2
if __name__=='__main__': raise SystemExit(main())
