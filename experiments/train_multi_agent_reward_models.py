from __future__ import annotations
import argparse,json
from pathlib import Path
from rlaif.preference_dataset import read_jsonl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preferences',required=True); ap.add_argument('--output',required=True); ap.add_argument('--agent',required=True); args=ap.parse_args()
    rows=read_jsonl(args.preferences); usable=[r for r in rows if r.get('usable_for_training') and r.get('resolved_original_winner') not in {'tie','abstain'}]
    if not usable: raise SystemExit('no usable non-tie/non-abstain labels')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps({'agent_type':args.agent,'smoke_placeholder':True,'usable':len(usable)}))
    print(json.dumps({'agent':args.agent,'usable':len(usable)}))
if __name__=='__main__': main()
