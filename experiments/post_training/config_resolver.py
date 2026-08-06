"""Resolve the nine seed-specific training jobs without mutating paper configs."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml
from . import METHODS, SEEDS
from .artifacts import sha256_file

JOB_COUNT=9

def training_jobs() -> list[dict[str,int|str]]:
    return [{"method_id":m,"seed":s,"phase":3 if m==METHODS[0] else 7 if m==METHODS[1] else 8} for m in METHODS for s in SEEDS]

def resolve_child_configs(base_checkpoints: dict[int,Path], base_manifests: dict[int,Path], reward_model: Path, templates: dict[str,dict[str,Any]]) -> list[dict[str,Any]]:
    jobs=[]
    for method in METHODS:
      for seed in SEEDS:
        cfg=deepcopy(templates[method]); cfg["method_id"]=method
        train=cfg.setdefault("training",{}); train["seed"]=seed
        init=train.setdefault("initialization",{})
        if method==METHODS[0]: init.update(mode="random",load_actors=False,load_critic=False,load_optimizers=False,strict=True)
        else:
          cp,manifest=Path(base_checkpoints[seed]),Path(base_manifests[seed])
          if not cp.is_file() or not manifest.is_file(): raise FileNotFoundError(f"base lineage missing for seed {seed}")
          init.update(mode="pretrained_policy",checkpoint_path=str(cp),checkpoint_sha256=sha256_file(cp),manifest_path=str(manifest),load_actors=True,load_critic=True,load_optimizers=False,strict=True)
        if method==METHODS[1]: cfg["rlaif"]={"enabled":False}
        elif method==METHODS[2]:
          if not reward_model.is_file(): raise FileNotFoundError(reward_model)
          cfg.setdefault("rlaif",{}).update(enabled=True,scope="assignment",fallback_to_env_reward=False,fail_on_invalid_reward_model=True,reward_model_sha256=sha256_file(reward_model))
        jobs.append(cfg)
    return jobs

def dump_resolved(config:dict[str,Any], path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(yaml.safe_dump(config,sort_keys=False))
