"""Dry-by-default Phase 0-11 plan for isolated formal post-training."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from .post_training import METHODS,SEEDS,OUTPUT_ROOT
PHASES=(
 "scenario-bank, GPU, and base artifact checks","code and environment verification","reward reference scales and base config resolution","train mappo_env_post_base seeds 1/2/3","collect assignment preferences from all three base checkpoints","AI labels, quality gate, and Bradley–Terry Reward Model","resolve children; fairness and parent-lineage gate","train mappo_env_post_continued seeds 1/2/3","train mappo_rlaif_assignment_post seeds 1/2/3","checkpoint readiness and final freeze","900-row common-scenario benchmark","paired statistical analysis")
def build_plan(through=11):
    if through not in range(12): raise ValueError("--through must be 0..11")
    return [{"phase":i,"description":PHASES[i],"execute":False} for i in range(through+1)]
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--through",type=int,default=11); p.add_argument("--device",default="cuda"); p.add_argument("--execute",action="store_true"); p.add_argument("--resume", action="store_true")
    a=p.parse_args(argv)
    if a.execute: raise SystemExit("Formal execution requires separately validated phase-specific commands; this orchestrator is dry-plan only")
    print("Isolated formal post-training dry plan")
    for x in build_plan(a.through): print(f"Phase {x['phase']}: {x['description']}")
    print("3 base/continuation/RLAIF methods")
    print("9 training jobs (3 methods × seeds 1/2/3)")
    print("900 benchmark rows (3 methods × 3 seeds × 100 common scenarios)")
    print("Bradley–Terry Reward Model")
    print("DPO not used")
    print("primary comparison:")
    print("mappo_rlaif_assignment_post")
    print("vs")
    print("mappo_env_post_continued")
    print(f"output root: {OUTPUT_ROOT}/<commit>/")
    print("DRY RUN: evaluator, RM/MAPPO training, checkpoints, and benchmark were not executed")
if __name__=="__main__": main()
