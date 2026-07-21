import json, subprocess
from pathlib import Path
from data_pipeline.build_instance import build_instance
from evaluation.result_schema import RESULT_FIELDS
from evaluation.runner import EvaluationRunner


def make_instance(tmp_path):
    built=build_instance("configs/shanghai_small.yaml",fallback=True,output_root=tmp_path/"instance")
    return Path(built["output_directory"])/"instance.json"

def test_runner_saves_episode_result(tmp_path):
    method={"name":"truck_only","assignment_policy":"truck_only","bus_policy":"no_charge","rlaif_enabled":False}
    runner=EvaluationRunner({"name":"test","instance_name":"fallback","output_dir":str(tmp_path/"out")},make_instance(tmp_path),method,0)
    row=runner.run_episode(); assert row["status"]=="success"; assert set(RESULT_FIELDS)<=set(row)
    assert (tmp_path/"out"/"raw"/"truck_only"/"seed_0.json").is_file()

def test_missing_checkpoints_are_skipped(tmp_path):
    instance=make_instance(tmp_path); config={"name":"test","output_dir":str(tmp_path/"out")}
    missing={"name":"assignment_ppo","assignment_policy":"assignment_ppo","checkpoint":str(tmp_path/"missing.pt"),"bus_policy":"no_charge","rlaif_enabled":False}
    assert EvaluationRunner(config,instance,missing,0).run_episode()["status"]=="skipped_missing_checkpoint"
    rlaif={**missing,"name":"assignment_ppo_rlaif","checkpoint":str(tmp_path/"policy.pt"),"rlaif_enabled":True,"reward_model_checkpoint":str(tmp_path/"reward_model.pt")}
    (tmp_path/"policy.pt").write_bytes(b"not loaded because reward checkpoint is absent")
    assert EvaluationRunner(config,instance,rlaif,0).run_episode()["status"]=="skipped_missing_checkpoint"

def test_results_are_git_ignored(): assert subprocess.run(["git","check-ignore","-q","results/probe.json"]).returncode==0


def test_runner_evaluates_valid_async_mappo_checkpoint(tmp_path):
    from training.mappo_trainer import train_mappo_async
    from utils.config import load_config
    config = load_config("configs/train_mappo_async.yaml")
    config["training"].update({"total_episodes": 1, "rollout_episodes": 1, "ppo_epochs": 1, "batch_size": 32})
    config["output"] = {"checkpoint_path": str(tmp_path/"mappo.pt"), "training_log_path": str(tmp_path/"train.csv"), "eval_path": str(tmp_path/"eval.json")}
    train_mappo_async(config, output_root=tmp_path/"train_instance")
    instance = make_instance(tmp_path)
    method={"name":"mappo_async","assignment_policy":"mappo_async","checkpoint":str(tmp_path/"mappo.pt"),"rlaif_enabled":False}
    row=EvaluationRunner({"name":"test","instance_name":"fallback","output_dir":str(tmp_path/"out")},instance,method,0).run_episode()
    assert row["status"]=="success"
