from __future__ import annotations
import argparse, json
from pathlib import Path
from rlaif.preference_dataset import read_jsonl, write_jsonl
from rlaif.preference_schema_v2 import validate_preference_state
from rlaif.pair_selector import select_informative_pairs, randomize_display_order, resolve_original_winner
from rlaif.agent_prompt_builders import build_agent_prompt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input'); ap.add_argument('--output',required=True); ap.add_argument('--offline',action='store_true'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--agent',default='assignment'); args=ap.parse_args()
    records=read_jsonl(args.input) if args.input and Path(args.input).exists() else []
    name=Path(__file__).stem
    out=[]
    if name.endswith('preference_prompts'):
        for st in records:
            validate_preference_state(st)
            for p in select_informative_pairs(st): out.append(build_agent_prompt(st, randomize_display_order(p, seed=args.seed)))
    elif name.endswith('preferences'):
        # Offline mode deliberately creates prompts/empty preference files only; no labels are fabricated.
        out=[] if args.offline else [resolve_original_winner(r) for r in records if r.get('label_source') in {'external_evaluator_api','validated_replay'}]
    else:
        out=records
    write_jsonl(args.output,out); print(json.dumps({'records':len(out),'offline':args.offline,'script':name}))
if __name__=='__main__': main()
