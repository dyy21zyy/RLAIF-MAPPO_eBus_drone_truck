from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal
import hashlib, json, time, csv, subprocess

ExperimentKind = Literal["ablation", "sensitivity"]
ExecutionMode = Literal["retrain_and_evaluate", "fixed_policy_evaluate"]
JOB_STATUSES = ("planned","validated","training_running","training_success","training_failed","evaluation_running","evaluation_success","evaluation_failed","aggregation_success","aggregation_failed","blocked_missing_artifact","blocked_invalid_configuration","blocked_incompatible_lineage")
TERMINAL_SUCCESS={"evaluation_success","aggregation_success"}

@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    display_name: str
    experiment_kind: ExperimentKind
    execution_mode: ExecutionMode
    base_training_config: Path | None
    base_benchmark_config: Path
    config_overrides: dict[str, Any]
    sensitivity_parameter: str | None = None
    sensitivity_value: Any | None = None
    sensitivity_protocol: str | None = None

@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    variant_id: str
    training_seed: int | None
    scenario_family_id: str
    train_bank_hash: str | None
    validation_bank_hash: str | None
    test_bank_hash: str
    policy_checkpoint_path: Path | None
    output_root: Path

def canonical_json_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()

def sha256_file(path: str|Path|None) -> str|None:
    if not path or not Path(path).is_file(): return None
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def make_job_identity(**fields: Any) -> dict[str, Any]:
    clean={k:v for k,v in fields.items() if v is not None}
    clean['identity_hash']=canonical_json_hash(clean)
    return clean

def should_resume_skip(previous: dict[str, Any]|None, identity: dict[str, Any]) -> bool:
    return bool(previous and previous.get('status') in TERMINAL_SUCCESS and previous.get('identity',{}).get('identity_hash') == identity.get('identity_hash'))

def run_subprocess(command: list[str], *, stage: str, cwd: str|Path|None=None) -> dict[str, Any]:
    start=time.time()
    cp=subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    rec={"stage":stage,"command":command,"return_code":cp.returncode,"stdout":cp.stdout,"stderr":cp.stderr,"start_time":start,"end_time":time.time(),"runtime":time.time()-start}
    if cp.returncode != 0:
        raise ExperimentSubprocessError(rec)
    return rec

class ExperimentSubprocessError(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        super().__init__(record.get('stderr') or record.get('stdout') or f"command failed: {record.get('command')}")
        self.record=record

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.open('a', encoding='utf-8').write(json.dumps(row, sort_keys=True, default=str)+'\n')

def write_job_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root/'job_results.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,default=str)+'\n' for r in rows), encoding='utf-8')
    fields=sorted({k for r in rows for k in r if not isinstance(r.get(k), (dict,list))})
    with (root/'job_results.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k) for k in fields} for r in rows])
    failures=[r for r in rows if str(r.get('status','')).endswith('_failed') or str(r.get('status','')).startswith('blocked_')]
    (root/'failure_report.json').write_text(json.dumps({'failure_count':len(failures),'failures':failures}, indent=2, sort_keys=True, default=str), encoding='utf-8')
