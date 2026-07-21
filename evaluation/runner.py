"""Generic fair, seed-controlled Stage 8 evaluation runner."""
from __future__ import annotations
import hashlib, importlib.util, json, time
from pathlib import Path
from typing import Any

from baselines import build_assignment_policy, build_bus_policy
from envs.delivery_env import DynamicDeliveryEnv
from envs.state_builder import build_candidate_action_features
from evaluation.metrics import collect_environment_metrics
from evaluation.reporting import write_episode, write_records_csv
from evaluation.result_schema import normalize_result
from rlaif.preference_dataset import ACTION_FEATURE_KEYS

LEARNED_ASSIGNMENT={"assignment_ppo","mappo_async"}


def _first_feasible(action_mask):
    return next((index for index, feasible in enumerate(action_mask) if bool(feasible)), 0)


def _is_bus_arrival_charge(observation, selected: int, env: DynamicDeliveryEnv) -> tuple[bool, float]:
    if observation["agent"] != "bus" or observation.get("event_type") != "BUS_ARRIVAL":
        return False, 0.0
    seconds = float(env.config["bus"]["charging_actions_sec"][selected])
    return seconds > 0.0, seconds

class EvaluationRunner:
    def __init__(self, config: dict[str,Any], instance, method: dict[str,Any], seed: int):
        self.config=config; self.instance=Path(instance); self.method=dict(method); self.seed=int(seed)
        self.output_dir=Path(config.get("output_dir","results/benchmark")); self.experiment_id=str(config.get("name","stage8_experiment"))

    def _base(self, status="success", error_message=""):
        payload=json.dumps({"experiment":self.config,"method":self.method},sort_keys=True,default=str).encode()
        return normalize_result({"experiment_id":self.experiment_id,"method_name":self.method.get("name",self.method.get("assignment_policy","unknown")),"seed":self.seed,"instance_name":self.config.get("instance_name",self.instance.parent.name),"config_hash":hashlib.sha256(payload).hexdigest()[:16],"rlaif_enabled":bool(self.method.get("rlaif_enabled",False)),"assignment_policy_name":self.method.get("assignment_policy",""),"bus_policy_name":self.method.get("bus_policy",""),"status":status,"error_message":error_message})

    def _record_path(self):
        safe=str(self.method.get("name","method")).replace("/","_")
        return self.output_dir/"raw"/safe/f"seed_{self.seed}.json"

    def _prerequisite_status(self):
        assignment=self.method.get("assignment_policy","")
        if "delayed_reward" in self.method:
            return "skipped_unsupported", "Delayed-reward ablation is declared but not supported by the current Stage 7 runtime."
        if assignment not in LEARNED_ASSIGNMENT: return None
        checkpoint=self.method.get("checkpoint")
        if not checkpoint or not Path(checkpoint).is_file(): return "skipped_missing_checkpoint",f"Required learned-policy checkpoint is missing: {checkpoint}"
        if self.method.get("rlaif_enabled"):
            reward=self.method.get("reward_model_checkpoint")
            if not reward or not Path(reward).is_file(): return "skipped_missing_checkpoint",f"rlaif_enabled=true requires a valid reward_model.pt: {reward}"
        if importlib.util.find_spec("torch") is None: return "skipped_missing_dependency","PyTorch is unavailable; learned policy was not substituted."
        return None

    def _load_policies(self):
        assignment_name=self.method.get("assignment_policy","truck_only")
        if assignment_name=="assignment_ppo":
            from training.ppo_trainer import load_assignment_checkpoint
            assignment,_=load_assignment_checkpoint(self.method["checkpoint"]); bus=build_bus_policy(self.method.get("bus_policy","no_charge")); return assignment,bus
        if assignment_name=="mappo_async":
            from training.mappo_trainer import load_checkpoint
            actors, _critic, _checkpoint = load_checkpoint(self.method["checkpoint"]); return actors, actors
        return build_assignment_policy(assignment_name,self.seed),build_bus_policy(self.method.get("bus_policy","no_charge"))

    def run_episode(self):
        prerequisite=self._prerequisite_status()
        if prerequisite:
            record=self._base(*prerequisite); write_episode(self._record_path(),record); return record
        started=time.perf_counter(); env_reward=0.0; rlaif_reward=0.0; charge_count=0; charge_energy=0.0; fallback_events=0
        try:
            env=DynamicDeliveryEnv(self.instance); assignment,bus=self._load_policies()
            reward_wrapper=None
            if self.method.get("rlaif_enabled"):
                from training.reward_model_wrapper import RewardModelWrapper
                reward_wrapper=RewardModelWrapper(self.method.get("reward_model_checkpoint"),enabled=True,validation=self.method.get("rlaif_validation",{}),fallback_to_env_reward=bool(self.method.get("fallback_to_env_reward", True)),fail_on_invalid_reward_model=bool(self.method.get("fail_on_invalid_reward_model", False)),reward_clip=self.method.get("reward_clip"))
            observation,_=env.reset(seed=self.seed)
            while observation["agent"]!="terminal":
                mask=[bool(v) for v in observation["action_mask"]]
                if observation["agent"]=="assignment":
                    before=list(observation["features"])
                    if self.method.get("assignment_policy") in LEARNED_ASSIGNMENT:
                        policy_mask = mask if self.method.get("action_mask", True) else [True] * len(mask)
                        selected=assignment["assignment"].act((before + [0.0] * assignment["assignment"].obs_dim)[:assignment["assignment"].obs_dim], observation["candidate_features"], policy_mask, deterministic=True)[0] if self.method.get("assignment_policy")=="mappo_async" else assignment.act(before,policy_mask,deterministic=True)[0]
                    else: selected=assignment.select_action(observation,env)
                    if not mask or not any(mask) or selected>=len(mask) or not mask[selected]: fallback_events+=1
                    action_features=None
                    if reward_wrapper is not None:
                        parcel=env.parcels[env.current_decision.event.payload["parcel_id"]]
                        features=build_candidate_action_features(env,parcel,selected,mask[selected])
                        action_features=[float(features[key]) for key in ACTION_FEATURE_KEYS]
                    observation,reward,*_=env.step(selected); env_reward+=float(reward)
                    if reward_wrapper is not None: rlaif_reward+=reward_wrapper.score(before,action_features,selected)
                else:
                    if self.method.get("assignment_policy")=="mappo_async":
                        policy_mask = mask if self.method.get("action_mask", True) else [True] * len(mask)
                        selected=bus[str(observation["agent_id"])].act((observation["features"] + [0.0] * bus[str(observation["agent_id"])].obs_dim)[:bus[str(observation["agent_id"])].obs_dim], observation["candidate_features"], policy_mask, deterministic=True)[0]
                    elif observation["agent"]=="bus" and observation.get("event_type")=="BUS_ARRIVAL":
                        selected=bus.select_action(observation,env)
                    else:
                        selected=_first_feasible(mask)
                    is_charge, seconds = _is_bus_arrival_charge(observation, selected, env)
                    if is_charge:
                        charge_count+=1; charge_energy+=float(env.config["bus"]["charging_power_kw"])*(seconds/3600.0)
                    observation,reward,*_=env.step(selected); env_reward+=float(reward)
            env.get_metrics()  # Stable Stage 3 metric hook; detailed Stage 8 metrics are derived below.
            row=self._base(); row.update(collect_environment_metrics(env,bus_charging_count=charge_count,bus_charging_energy=charge_energy,fallback_events=fallback_events))
            weight=float(self.method.get("lambda_rlaif",1.0)); row.update({"total_env_reward":env_reward,"total_rlaif_reward":rlaif_reward,"episode_reward":env_reward+weight*rlaif_reward,"runtime_seconds":time.perf_counter()-started})
            row=normalize_result(row)
        except Exception as exc:
            row=self._base("failed",f"{type(exc).__name__}: {exc}"); row["runtime_seconds"]=time.perf_counter()-started
        write_episode(self._record_path(),row); return row

    def run_many(self,seeds):
        rows=[]
        for seed in seeds:
            rows.append(EvaluationRunner(self.config,self.instance,self.method,int(seed)).run_episode())
            if self.config.get("fail_fast") and rows[-1]["status"]=="failed": break
        write_records_csv(self.output_dir/"episodes.csv",rows)
        return rows
