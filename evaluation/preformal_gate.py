"""Auditable pre-formal experiment gate orchestration."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import yaml

PREFORMAL_STAGES = (
    "repository_verification",
    "artifact_inventory",
    "scenario_bank_preparation",
    "scenario_bank_validation",
    "reward_scale_estimation",
    "reward_scale_validation",
    "reward_model_validation",
    "environment_mappo_training",
    "assignment_ppo_training",
    "assignment_rlaif_mappo_training",
    "full_rlaif_mappo_training",
    "policy_checkpoint_validation",
    "benchmark_execution",
    "ablation_execution",
    "sensitivity_execution",
    "paired_result_validation",
    "formal_launch_plan",
)

STAGE_STATUSES = frozenset({
    "not_started", "running", "passed", "failed", "blocked_dependency",
    "blocked_missing_artifact", "blocked_invalid_artifact", "skipped_optional",
})

OVERALL_STATUSES = frozenset({
    "PREFORMAL_DIAGNOSTIC_PASSED", "PREFORMAL_DIAGNOSTIC_FAILED",
    "PREFORMAL_ENVIRONMENT_PATH_PASSED_RLAIF_BLOCKED",
    "PREFORMAL_ASSIGNMENT_RLAIF_PASSED_FULL_RLAIF_BLOCKED",
    "PREFORMAL_ALL_REQUIRED_PATHS_PASSED", "BLOCKED_SCENARIO_BANK",
    "BLOCKED_REWARD_SCALE", "BLOCKED_REWARD_MODELS", "BLOCKED_POLICY_TRAINING",
    "BLOCKED_BENCHMARK", "BLOCKED_ABLATION", "BLOCKED_SENSITIVITY",
    "BLOCKED_FULL_TEST_SUITE",
})

DEFAULT_DEPENDENCIES = {
    "artifact_inventory": ("repository_verification",),
    "scenario_bank_preparation": ("artifact_inventory",),
    "scenario_bank_validation": ("scenario_bank_preparation",),
    "reward_scale_estimation": ("scenario_bank_validation",),
    "reward_scale_validation": ("reward_scale_estimation",),
    "reward_model_validation": ("reward_scale_validation",),
    "environment_mappo_training": ("reward_model_validation",),
    "assignment_ppo_training": ("environment_mappo_training",),
    "assignment_rlaif_mappo_training": ("assignment_ppo_training",),
    "full_rlaif_mappo_training": ("assignment_rlaif_mappo_training",),
    "policy_checkpoint_validation": ("environment_mappo_training",),
    "benchmark_execution": ("policy_checkpoint_validation",),
    "ablation_execution": ("benchmark_execution",),
    "sensitivity_execution": ("ablation_execution",),
    "paired_result_validation": ("benchmark_execution",),
    "formal_launch_plan": ("paired_result_validation", "sensitivity_execution"),
}

OPTIONAL_STRICT_STAGES = {"full_rlaif_mappo_training"}
NOTE = "These are pre-formal validation runs. They are not final paper experiments."

@dataclass
class StageRecord:
    stage_id: str
    required: bool
    dependencies: list[str]
    status: str = "not_started"
    start_time: float | None = None
    end_time: float | None = None
    runtime: float | None = None
    invoked: str | None = None
    input_artifact_hashes: dict[str, str | None] = field(default_factory=dict)
    output_artifact_hashes: dict[str, str | None] = field(default_factory=dict)
    failure_reason: str | None = None
    exception_type: str | None = None

    def set_status(self, status: str) -> None:
        if status not in STAGE_STATUSES: raise ValueError(f"invalid stage status {status}")
        self.status = status


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def ensure_runtime_root(path: Path, mode: str) -> None:
    allowed = [Path("results/diagnostic"), Path("results/preformal"), Path("results/formal")]
    rel = Path(os.path.relpath(path.resolve(), Path.cwd().resolve())) if path.is_absolute() else path
    path_text = str(path)
    in_pytest_tmp = "pytest-" in path_text and "/results/" in path_text
    if not any(str(rel).startswith(str(a)) for a in allowed) and not in_pytest_tmp:
        raise ValueError(f"runtime artifacts must be under {', '.join(map(str, allowed))}: {path}")
    if mode == "diagnostic" and not (str(rel).startswith("results/diagnostic") or "/results/diagnostic/" in path_text):
        raise ValueError("diagnostic mode must write under results/diagnostic")
    path.mkdir(parents=True, exist_ok=True)


def _run_command(command: list[str], output: Path) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    payload = {"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "runtime": time.time()-start}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")
    return payload

class PreformalGate:
    def __init__(self, config: dict[str, Any], *, stage_functions: dict[str, Callable[[StageRecord], Any]] | None = None):
        self.config = config
        self.mode = str(config.get("run_classification"))
        self.stage = str(config.get("experiment_stage"))
        if config.get("publication_eligible") is not False: raise ValueError("pre-formal runs must set publication_eligible=false")
        if self.mode not in {"diagnostic", "formal"}: raise ValueError("run_classification must be diagnostic or formal")
        if self.mode == "diagnostic" and self.stage != "preformal_diagnostic": raise ValueError("diagnostic gate requires experiment_stage=preformal_diagnostic")
        if self.mode == "formal" and self.stage != "preformal": raise ValueError("strict gate requires experiment_stage=preformal")
        self.output_root = Path(config.get("output_root", "results/diagnostic/preformal_gate" if self.mode == "diagnostic" else "results/preformal/gate"))
        ensure_runtime_root(self.output_root, self.mode)
        self.stage_functions = stage_functions or {}
        self.records = {s: StageRecord(s, self._required(s), list(DEFAULT_DEPENDENCIES.get(s, ()))) for s in PREFORMAL_STAGES}

    def _required(self, stage: str) -> bool:
        return not (self.mode == "formal" and stage in OPTIONAL_STRICT_STAGES and not self.config.get("enable_full_rlaif", False))

    def _default_stage(self, rec: StageRecord) -> None:
        commands = self.config.get("commands", {}).get(rec.stage_id)
        if commands:
            rec.invoked = " ".join(commands)
            _run_command([str(x) for x in commands], self.output_root / f"{rec.stage_id}.command.json")
        else:
            marker = self.output_root / f"{rec.stage_id}.json"
            marker.write_text(json.dumps({"stage_id": rec.stage_id, "run_classification": self.mode, "publication_eligible": False, "note": NOTE}, sort_keys=True), encoding="utf-8")
            rec.invoked = "preformal_gate_builtin_marker"
            rec.output_artifact_hashes[str(marker)] = sha256_file(marker)

    def _strict_artifact_gate(self, rec: StageRecord) -> bool:
        for path in self.config.get("formal_candidate_artifacts", {}).get(rec.stage_id, []):
            p = Path(path)
            if not p.exists(): rec.set_status("blocked_missing_artifact"); rec.failure_reason = f"missing artifact: {p}"; return False
            h = sha256_file(p)
            if not h or h.startswith("0"*8): rec.set_status("blocked_invalid_artifact"); rec.failure_reason = f"invalid artifact hash: {p}"; return False
            text = p.read_text(errors="ignore") if p.is_file() and p.stat().st_size < 2_000_000 else ""
            if self.mode == "formal" and ("diagnostic" in text or "smoke" in text or "REPLACE_WITH_REAL_HASH" in text):
                rec.set_status("blocked_invalid_artifact"); rec.failure_reason = f"non-formal or placeholder artifact rejected: {p}"; return False
            rec.input_artifact_hashes[str(p)] = h
        return True

    def run(self) -> dict[str, Any]:
        for sid in PREFORMAL_STAGES:
            rec = self.records[sid]
            if not rec.required:
                rec.set_status("skipped_optional"); continue
            bad = [d for d in rec.dependencies if self.records[d].required and self.records[d].status != "passed"]
            if bad:
                rec.set_status("blocked_dependency"); rec.failure_reason = "blocked by: " + ", ".join(bad); continue
            if not self._strict_artifact_gate(rec): continue
            rec.start_time = time.time(); rec.set_status("running")
            try:
                (self.stage_functions.get(sid) or self._default_stage)(rec)
                if rec.status == "running": rec.set_status("passed")
            except Exception as exc:
                rec.set_status("failed"); rec.failure_reason = str(exc); rec.exception_type = type(exc).__name__
            finally:
                rec.end_time = time.time(); rec.runtime = rec.end_time - rec.start_time
        status = self.overall_status()
        report = {"overall_status": status, "run_classification": self.mode, "experiment_stage": self.stage, "publication_eligible": False, "note": NOTE, "stages": [asdict(self.records[s]) for s in PREFORMAL_STAGES]}
        (self.output_root/"preformal_gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report

    def overall_status(self) -> str:
        passed = {s for s,r in self.records.items() if r.status in {"passed", "skipped_optional"}}
        if self.mode == "diagnostic":
            return "PREFORMAL_DIAGNOSTIC_PASSED" if all(self.records[s].status == "passed" for s in PREFORMAL_STAGES) else "PREFORMAL_DIAGNOSTIC_FAILED"
        if self.records["scenario_bank_validation"].status != "passed": return "BLOCKED_SCENARIO_BANK"
        if self.records["reward_scale_validation"].status != "passed": return "BLOCKED_REWARD_SCALE"
        if self.records["reward_model_validation"].status != "passed": return "BLOCKED_REWARD_MODELS"
        if self.records["environment_mappo_training"].status == "passed" and self.records["assignment_rlaif_mappo_training"].status != "passed": return "PREFORMAL_ENVIRONMENT_PATH_PASSED_RLAIF_BLOCKED"
        if self.records["assignment_rlaif_mappo_training"].status == "passed" and self.records["full_rlaif_mappo_training"].status != "passed": return "PREFORMAL_ASSIGNMENT_RLAIF_PASSED_FULL_RLAIF_BLOCKED"
        for s in ("policy_checkpoint_validation","benchmark_execution","ablation_execution","sensitivity_execution","paired_result_validation","formal_launch_plan"):
            if self.records[s].required and self.records[s].status != "passed":
                return {"benchmark_execution":"BLOCKED_BENCHMARK","ablation_execution":"BLOCKED_ABLATION","sensitivity_execution":"BLOCKED_SENSITIVITY"}.get(s,"BLOCKED_FULL_TEST_SUITE")
        return "PREFORMAL_ALL_REQUIRED_PATHS_PASSED"

def run_config(config_path: str | Path) -> dict[str, Any]:
    return PreformalGate(load_config(config_path)).run()
