"""Small audited command runner used by pre-formal orchestration."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

def run_logged(command: list[str], log_path: str | Path, *, cwd: str | Path | None = None) -> dict[str, object]:
    start=time.time(); proc=subprocess.run(command,cwd=cwd,text=True,capture_output=True,check=False)
    payload={'command':command,'cwd':str(cwd) if cwd else None,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'runtime':time.time()-start}
    Path(log_path).parent.mkdir(parents=True,exist_ok=True); Path(log_path).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    if proc.returncode: raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")
    return payload
