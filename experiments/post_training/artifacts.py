"""Content-addressed artifacts and resumable phase markers."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Iterable

FORMAL_ROOT = Path("results/formal_post_training")
FORBIDDEN_ROOT = Path("results/formal")

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def run_root(commit: str) -> Path:
    if not commit or "/" in commit or commit in {".",".."}: raise ValueError("invalid commit")
    root=FORMAL_ROOT/commit
    if root == FORBIDDEN_ROOT or FORBIDDEN_ROOT in root.parents: raise ValueError("post-training output overlaps original formal root")
    return root

def input_hashes(paths: Iterable[str | Path]) -> dict[str,str]:
    result={}
    for item in paths:
        p=Path(item)
        if not p.is_file(): raise FileNotFoundError(p)
        result[str(p)]=sha256_file(p)
    return result

def write_phase_marker(root: Path, phase: int, inputs: Iterable[str | Path], outputs: Iterable[str | Path]=()) -> Path:
    if phase not in range(12): raise ValueError("phase must be 0..11")
    payload={"schema_version":1,"phase":phase,"input_sha256":input_hashes(inputs),"output_sha256":input_hashes(outputs)}
    path=root/"phase_markers"/f"phase_{phase}.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    return path

def marker_is_current(path: Path) -> bool:
    try: data=json.loads(path.read_text())
    except (OSError,ValueError): return False
    return all(Path(p).is_file() and sha256_file(p)==digest for group in ("input_sha256","output_sha256") for p,digest in data.get(group,{}).items())
