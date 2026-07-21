"""Dependency-light Stage 8 experiment-framework smoke test."""
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
from experiments.run_benchmark import run_config
from rlaif.torch_runtime import is_torch_runtime_available

METHODS=[
 {"name":"truck_only","assignment_policy":"truck_only","bus_policy":"no_charge","rlaif_enabled":False},
 {"name":"random_feasible","assignment_policy":"random_feasible","bus_policy":"uniform_30","rlaif_enabled":False},
 {"name":"bus_drone_only","assignment_policy":"bus_drone_only","bus_policy":"uniform_30","rlaif_enabled":False},
 {"name":"truck_drone","assignment_policy":"truck_drone","bus_policy":"uniform_30","rlaif_enabled":False},
 {"name":"rule_based","assignment_policy":"rule_based","bus_policy":"battery_threshold","rlaif_enabled":False},
 {"name":"assignment_ppo","assignment_policy":"assignment_ppo","checkpoint":"results/checkpoints/missing_assignment_ppo.pt","bus_policy":"uniform_30","rlaif_enabled":False},
]

def main():
    with tempfile.TemporaryDirectory(prefix="stage8_smoke_") as temp:
        root=Path(temp); output=root/"results"
        config={"experiment":{"name":"stage8_smoke","instance_name":"fallback_smoke","base_config":"configs/shanghai_small.yaml","seeds":[0],"output_dir":str(output),"max_episodes_per_method":1,"fail_fast":False,"skip_missing_checkpoints":True,"smoke":True,"fallback":True},"methods":METHODS}
        rows=run_config(config,temporary_root=root/"instance")
        expected={m["name"] for m in METHODS[:5]}; successful={r["method_name"] for r in rows if r["status"]=="success"}
        assert expected<=successful, (expected,successful)
        learned=next(r for r in rows if r["method_name"]=="assignment_ppo"); assert learned["status"]=="skipped_missing_checkpoint"
        assert (output/"summary"/"summary_metrics.csv").is_file(); assert (output/"summary"/"summary_metrics.json").is_file(); assert (output/"summary"/"method_status.csv").is_file()
        ignored=subprocess.run(["git","check-ignore","-q","results/stage8_probe.json"],check=False).returncode==0
        assert ignored,"results/ must be ignored by Git"
        print(f"Stage 8 smoke test passed: {len(successful)} baselines; torch_available={is_torch_runtime_available()}; learned_status={learned['status']}")
    return 0
if __name__=="__main__": raise SystemExit(main())
