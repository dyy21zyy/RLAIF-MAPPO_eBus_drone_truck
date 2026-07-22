from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from training.reward_model_wrapper import load_strict_agent_reward_checkpoint
from rlaif.reward_model_dataset import build_reward_pair_dataset
from rlaif.reward_model_normalization import FeatureNormalization
from rlaif.reward_model_trainer import evaluate_agent_reward_model

def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument('--preferences',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',default='test'); ns=ap.parse_args(argv)
 ck,model=load_strict_agent_reward_checkpoint(ns.checkpoint,agent_type=torch.load(ns.checkpoint,map_location='cpu',weights_only=False)['agent_type'],formal=False)
 rows=[json.loads(l) for l in Path(ns.preferences).read_text().splitlines() if l.strip()]; ds=build_reward_pair_dataset(rows,agent_type=ck['agent_type'],formal_mode=False)
 sn=FeatureNormalization(tuple(ck['state_normalization_mean']),tuple(ck['state_normalization_std']),tuple(ck['state_feature_names'])); cn=FeatureNormalization(tuple(ck['candidate_normalization_mean']),tuple(ck['candidate_normalization_std']),tuple(ck['candidate_feature_names']))
 m=evaluate_agent_reward_model(model,ds,sn,cn); Path(ns.checkpoint).with_suffix('.evaluation.json').write_text(json.dumps({'split':ns.split,'metrics':m},indent=2)); print(json.dumps(m)); return 0 if ck.get('run_classification')!='formal' or ck.get('validation_status')=='passed' else 2
if __name__=='__main__': raise SystemExit(main())
