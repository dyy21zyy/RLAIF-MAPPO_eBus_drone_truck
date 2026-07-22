from __future__ import annotations
import argparse, json
from envs.reward_scales import load_reward_scale_artifact
from evaluation.scenario_bank import load_bank_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--scenario-bank", required=True); ap.add_argument("--run-classification", default="diagnostic")
    a=ap.parse_args(); bank=load_bank_manifest(a.scenario_bank); art=load_reward_scale_artifact(a.artifact, expected_training_bank_hash=bank.get("bank_hash"), formal_mode=(a.run_classification=="formal"))
    print(json.dumps({"validation_status":"passed","artifact_hash":art.artifact_hash,"training_scenario_bank_hash":art.training_scenario_bank_hash}, indent=2, sort_keys=True))
if __name__=="__main__": main()
