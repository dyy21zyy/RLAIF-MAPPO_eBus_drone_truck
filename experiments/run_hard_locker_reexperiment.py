"""Root-isolated, fail-closed plan for the hard-locker formal rerun."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class PhaseSpec:
    phase: int
    name: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[Path, ...]
    capture: Path | None = None


def build_plan(*, output_root: Path, commit: str) -> list[PhaseSpec]:
    """Build the complete plan; every writable path is rooted below output_root."""
    py = sys.executable
    root = output_root
    pref = root / "preferences"
    reward = root / "reward_models" / "reward_assignment.pt"
    scale = root / "reward_scales" / "final_reward_reference_scales.json"
    resolved = root / "resolved_configs"
    benchmark = root / "benchmark"
    phases = [
        PhaseSpec(1, "verification", ((py,"-m","pytest","-q","tests/test_hard_locker_capacity.py"),(py,"-m","experiments.smoke_test_environment","--config","configs/shanghai_small.yaml")), (root/"verification",)),
        PhaseSpec(2, "preferences_and_reward_model", ((py,"-m","experiments.generate_formal_multiagent_preferences","--config","configs/paper/rlaif_preference_generation.yaml","--output-root",str(pref)),(py,"-m","experiments.train_multi_agent_reward_models","--preferences",str(pref),"--config",str(resolved/"train_reward_assignment.yaml"),"--agent","assignment","--output",str(reward))), (pref,reward,resolved/"train_reward_assignment.yaml")),
        PhaseSpec(3, "reward_scales", ((py,"-m","experiments.estimate_reward_reference_scales","--scenario-bank","results/formal/scenario_banks/train/manifest.json","--config","configs/paper/reward_scale_estimation.yaml","--output",str(scale)),), (scale,)),
        PhaseSpec(4, "freeze", ((py,"-m","experiments.freeze_final_experiment_parameters","--config",str(resolved/"final_experiment_freeze.template.yaml"),"--output",str(root/"final_experiment_freeze.yaml")),), (resolved/"final_experiment_freeze.template.yaml",root/"final_experiment_freeze.yaml")),
        PhaseSpec(5, "mappo_env_training", tuple((py,"-m","experiments.train_mappo_async","--config",str(resolved/"train_mappo_env.yaml"),"--seed",str(seed),"--output-root",str(root/"mappo_env"/f"seed_{seed}")) for seed in (1,2,3)), tuple(root/"mappo_env"/f"seed_{seed}" for seed in (1,2,3))+(resolved/"train_mappo_env.yaml",)),
        PhaseSpec(6, "rlaif_mappo_training", tuple((py,"-m","experiments.train_mappo_async","--config",str(resolved/"train_mappo_rlaif_assignment.yaml"),"--seed",str(seed),"--output-root",str(root/"mappo_rlaif_assignment"/f"seed_{seed}")) for seed in (1,2,3)), tuple(root/"mappo_rlaif_assignment"/f"seed_{seed}" for seed in (1,2,3))+(resolved/"train_mappo_rlaif_assignment.yaml",)),
        PhaseSpec(7, "readiness", ((py,"-m","experiments.validate_formal_experiment_readiness","--config",str(resolved/"benchmark.yaml"),"--strict"),), (resolved/"benchmark.yaml",root/"readiness.json"), root/"readiness.json"),
        PhaseSpec(8, "benchmark_600_rows", ((py,"-m","experiments.run_paper_benchmark","--config",str(resolved/"benchmark.yaml"),"--output-root",str(benchmark)),), (benchmark,)),
        PhaseSpec(9, "paired_analysis", ((py,"-m","experiments.aggregate_paper_results","--input",str(benchmark),"--output",str(root/"paired_analysis.json")),), (root/"paired_analysis.json",)),
    ]
    return phases


def _under(root: Path, path: Path) -> bool:
    try: path.resolve().relative_to(root.resolve()); return True
    except ValueError: return False


def validate_plan(plan: Sequence[PhaseSpec], root: Path) -> None:
    for phase in plan:
        for output in phase.outputs:
            if ".." in output.parts or not _under(root, output):
                raise ValueError(f"phase {phase.phase} output escapes run root: {output}")


def plan_hash(plan: Sequence[PhaseSpec]) -> str:
    payload = json.dumps([asdict(p) for p in plan], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _materialize_resolved_configs(root: Path) -> None:
    dest = root / "resolved_configs"; dest.mkdir(parents=True, exist_ok=True)
    replacements = {
        "results/formal/reward_scales/final_reward_reference_scales.json": str(root/"reward_scales"/"final_reward_reference_scales.json"),
        "results/formal/reward_models/reward_assignment.pt": str(root/"reward_models"/"reward_assignment.pt"),
        "results/formal/mappo_env": str(root/"mappo_env"),
        "results/formal/mappo_rlaif_assignment": str(root/"mappo_rlaif_assignment"),
        "checkpoints/formal/mappo_env_seed_": str(root/"mappo_env"/"seed_"),
        "checkpoints/formal/mappo_rlaif_assignment_seed_": str(root/"mappo_rlaif_assignment"/"seed_"),
        "checkpoints/formal/reward_assignment.pt": str(root/"reward_models"/"reward_assignment.pt"),
    }
    for name in ("train_reward_assignment.yaml","final_experiment_freeze.template.yaml","train_mappo_env.yaml","train_mappo_rlaif_assignment.yaml","benchmark.yaml"):
        text = (Path("configs/paper")/name).read_text()
        for old,new in replacements.items(): text=text.replace(old,new)
        (dest/name).write_text(text)


def main(argv: Sequence[str] | None = None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--through",type=int,choices=range(0,10),default=1); parser.add_argument("--resume",action="store_true"); parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    if _git("status","--porcelain"): raise SystemExit("hard-locker rerun requires a clean repository")
    commit=_git("rev-parse","HEAD"); root=Path("results/formal")/f"hard_locker_rerun_{commit[:12]}"
    plan=build_plan(output_root=root,commit=commit); validate_plan(plan,root); digest=plan_hash(plan)
    provenance={"commit":commit,"plan_hash":digest,"python":sys.version,"platform":platform.platform(),"reward_model_lineage":"regenerate: assignment schema v4 semantics changed"}
    if root.exists():
        if not args.resume: raise SystemExit(f"refusing to overwrite {root}; use --resume")
        stored=json.loads((root/"provenance.json").read_text())
        if stored.get("commit") != commit: raise SystemExit("resume provenance commit does not match HEAD")
        if stored.get("plan_hash") != digest: raise SystemExit("resume plan hash does not match current plan")
    selected=[p for p in plan if p.phase<=args.through]
    print(json.dumps({"output_root":str(root),"provenance":provenance,"plan":[asdict(p) for p in selected],"training_jobs":6,"benchmark_rows":600,"marker":"HARD_LOCKER_RERUN_PLAN_ISOLATED"},indent=2,default=str))
    if not args.execute: return
    root.mkdir(parents=True,exist_ok=args.resume); _materialize_resolved_configs(root); (root/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    for phase in selected:
        marker=root/f"phase_{phase.phase}.complete"
        if args.resume and marker.exists(): continue
        for command in phase.commands:
            result=subprocess.run(command,check=True,text=True,capture_output=phase.capture is not None)
            if phase.capture: phase.capture.parent.mkdir(parents=True,exist_ok=True); phase.capture.write_text(result.stdout)
        marker.write_text(hashlib.sha256(json.dumps(asdict(phase),sort_keys=True,default=str).encode()).hexdigest()+"\n")

if __name__=="__main__": main()
