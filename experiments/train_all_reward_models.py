from __future__ import annotations
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import torch
AGENTS=('assignment','truck','bus','station')
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
def build_reward_model_matrix_manifest(agent_results):
    ready=all(r.get('validation_status')=='passed' and r.get('run_classification','formal')=='formal' for r in agent_results.values())
    return {'agents':agent_results,'all_four_formal_models_ready':ready}
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--preferences',required=True); ap.add_argument('--config-dir',required=True); ap.add_argument('--output-dir',required=True); ns=ap.parse_args(argv); out=Path(ns.output_dir); out.mkdir(parents=True,exist_ok=True); results={}
    from experiments.train_multi_agent_reward_models import main as train_main
    for a in AGENTS:
        ck=out/f'reward_{a}.pt'; cfg=Path(ns.config_dir)/f'train_reward_{a}.yaml'; rc=train_main(['--preferences',ns.preferences,'--config',str(cfg),'--agent',a,'--output',str(ck)])
        status='missing'
        if ck.exists():
            data=torch.load(ck,map_location='cpu',weights_only=False); status=data.get('validation_status'); results[a]={'config_hash':fsha(cfg),'checkpoint_path':str(ck),'checkpoint_hash':fsha(ck),'validation_status':status,'run_classification':data.get('run_classification'),'best_epoch':data.get('best_epoch'),'validation_accuracy':data.get('validation_metrics',{}).get('pairwise_accuracy'),'test_accuracy':data.get('test_metrics',{}).get('pairwise_accuracy'),'event_coverage':list(data.get('per_event_metrics',{}))}
        else: results[a]={'config_hash':fsha(cfg),'checkpoint_path':str(ck),'validation_status':status}
    manifest=build_reward_model_matrix_manifest(results); (out/'reward_model_matrix_manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest)); return 0
if __name__=='__main__': raise SystemExit(main())
