"""Pre-formal artifact inventory and validation gates.

The helpers in this module intentionally classify artifacts from their internal
metadata and content hashes.  Directory names such as ``results/formal`` are
never used as evidence that an artifact is formal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math

PLACEHOLDER_VALUES = {"", "placeholder", "tbd", "unknown", "replace_with_real_hash", "replace_with_real_scale_hash", "replace_with_real_train_bank_hash"}
CLASSIFICATIONS = {"missing", "placeholder", "smoke", "diagnostic", "formal", "incompatible", "valid"}
REQUIRED_ARTIFACT_IDS = (
    "train_scenario_bank", "validation_scenario_bank", "test_scenario_bank",
    "reward_scale_artifact",
    "assignment_reward_checkpoint", "truck_reward_checkpoint", "bus_reward_checkpoint", "station_reward_checkpoint",
    "assignment_ppo_checkpoint", "mappo_env_checkpoint", "mappo_rlaif_assignment_checkpoint", "mappo_rlaif_all_checkpoint",
)

@dataclass(frozen=True)
class ArtifactInventoryItem:
    artifact_id: str
    artifact_type: str
    path: str | None
    exists: bool
    file_hash: str | None
    schema_version: int | str | None
    run_classification: str | None
    validation_status: str
    lineage: dict[str, Any]
    compatibility_status: str
    reason: str | None

class ArtifactValidationError(RuntimeError): pass

def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file(): return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def canonical_json_hash(payload: dict[str, Any], *, drop: Iterable[str] = ("artifact_hash", "file_hash")) -> str:
    clean = json.loads(json.dumps(payload, sort_keys=True, default=str))
    for k in drop: clean.pop(k, None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _is_placeholder(v: Any) -> bool:
    if v is None: return True
    s = str(v).strip().lower()
    return s in PLACEHOLDER_VALUES or s.startswith("replace_with_real_") or s.startswith("<") and s.endswith(">")

def _load_metadata(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8")), None
        try:
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)
            return (data if isinstance(data, dict) else {}), None
        except Exception as exc:
            return {}, f"unable to load artifact metadata: {exc}"
    except Exception as exc:
        return {}, f"unable to parse artifact metadata: {exc}"

def _configured_artifacts(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    src = config.get("artifacts") or config.get("artifact_inventory", {}).get("artifacts") or {}
    out: dict[str, dict[str, Any]] = {}
    for aid in REQUIRED_ARTIFACT_IDS:
        val = src.get(aid, {}) if isinstance(src, dict) else {}
        out[aid] = val if isinstance(val, dict) else {"path": val}
    return out

def build_artifact_inventory(config: dict[str, Any], *, strict: bool) -> list[ArtifactInventoryItem]:
    permitted = set(config.get("diagnostic", {}).get("permitted_artifacts", [])) | set(config.get("permitted_diagnostic_artifacts", []))
    expected_schema = config.get("expected_schema_versions", {})
    items: list[ArtifactInventoryItem] = []
    for aid, spec in _configured_artifacts(config).items():
        path_s = spec.get("path") or spec.get("manifest_path") or spec.get("checkpoint")
        atype = spec.get("artifact_type") or aid
        expected_hash = spec.get("hash") or spec.get("file_hash") or spec.get("artifact_hash")
        reason = None; meta: dict[str, Any] = {}; exists = False; fhash = None
        classification: str | None = None; validation = "failed"; compat = "incompatible"
        if _is_placeholder(path_s):
            classification = "placeholder"; reason = "placeholder path"
        else:
            p = Path(str(path_s)); exists = p.is_file(); fhash = sha256_file(p)
            if not exists:
                classification = "missing"; reason = "missing artifact"
            else:
                meta, load_reason = _load_metadata(p)
                internal = str(meta.get("run_classification") or meta.get("classification") or "").lower()
                classification = internal if internal in CLASSIFICATIONS else "incompatible"
                if load_reason: reason = load_reason
                if _is_placeholder(expected_hash) and expected_hash is not None:
                    classification = "placeholder"; reason = "placeholder hash"
                elif expected_hash is not None and str(expected_hash) != str(meta.get("artifact_hash", fhash)) and str(expected_hash) != fhash:
                    reason = "file hash mismatch"
                elif meta.get("artifact_hash") and str(meta.get("artifact_hash")) not in {str(fhash), canonical_json_hash(meta)}:
                    reason = "artifact hash mismatch"
                elif expected_schema.get(aid) is not None and meta.get("schema_version", meta.get("checkpoint_schema_version", meta.get("artifact_version"))) != expected_schema[aid]:
                    reason = "schema mismatch"
                elif meta.get("validation_status") not in {"passed", "valid", True}:
                    reason = "failed validation status"
                elif strict and classification != "formal":
                    reason = f"strict mode rejects {classification} artifact"
                elif (not strict) and classification in {"diagnostic", "smoke"} and aid not in permitted and "*" not in permitted:
                    reason = f"diagnostic artifact {aid} is not explicitly permitted"
                else:
                    validation = "passed"; compat = "compatible"; reason = None
        lineage = {
            "training_data": meta.get("training_data_lineage") or meta.get("training_data_hash"),
            "scenario_bank": meta.get("scenario_bank_lineage") or meta.get("training_scenario_bank_hash") or meta.get("train_bank_hash"),
            "reward_scale": meta.get("reward_scale_lineage") or meta.get("reward_scale_artifact_hash"),
            "reward_model": meta.get("reward_model_lineage") or meta.get("reward_checkpoint_hashes"),
        }
        items.append(ArtifactInventoryItem(aid, atype, None if path_s is None else str(path_s), exists, fhash, meta.get("schema_version", meta.get("checkpoint_schema_version", meta.get("artifact_version"))), classification, validation, lineage, compat, reason))
    return items

def write_artifact_inventory(config: dict[str, Any], output: str | Path, *, strict: bool) -> list[ArtifactInventoryItem]:
    items = build_artifact_inventory(config, strict=strict)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"artifacts": [asdict(i) for i in items]}, indent=2, sort_keys=True), encoding="utf-8")
    return items

# Backwards compatible lightweight inventory API used by earlier tests.
def inventory(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    return [{"path": str(Path(p)), "exists": Path(p).exists(), "is_file": Path(p).is_file(), "suffix": Path(p).suffix, "sha256": sha256_file(p), "binary_forbidden": Path(p).suffix in {'.npy','.npz','.pt','.pth','.pkl','.pickle','.bin','.onnx','.joblib'}} for p in paths]

def write_inventory(paths: Iterable[str | Path], output: str | Path) -> list[dict[str, object]]:
    rows = inventory(paths); Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps({"artifacts": rows}, indent=2, sort_keys=True), encoding="utf-8"); return rows
