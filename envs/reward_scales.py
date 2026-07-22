from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math
from typing import Iterable
from envs.reward_components import REWARD_COMPONENTS

PLACEHOLDERS = {"freeze-after-estimation", "tbd", "unknown", "placeholder", "", "replace_with_real_scale_hash", "replace_with_real_train_bank_hash"}

@dataclass(frozen=True)
class RewardScaleArtifact:
    artifact_version: int
    artifact_hash: str
    scales: dict[str, float]
    source_path: Path
    run_classification: str | None = None
    training_scenario_bank_hash: str | None = None
    estimator: dict | None = None

def canonical_payload_hash(payload: dict) -> str:
    clean = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clean.pop("artifact_hash", None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _placeholder(v) -> bool:
    return v is None or str(v).strip().lower() in PLACEHOLDERS or str(v).startswith("REPLACE_WITH_REAL_")

def load_reward_scale_artifact(path: str | Path, *, expected_hash: str | None = None,
                               expected_training_bank_hash: str | None = None,
                               required_components: Iterable[str] = REWARD_COMPONENTS,
                               formal_mode: bool = False) -> RewardScaleArtifact:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reward scale artifact not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reward scale JSON: {p}") from exc
    if payload.get("artifact_type", "reward_reference_scales") != "reward_reference_scales":
        raise ValueError("Reward scale artifact has wrong artifact_type")
    version = int(payload.get("artifact_version", -1))
    if version != 1: raise ValueError(f"Unsupported reward scale artifact_version: {version}")
    stored = str(payload.get("artifact_hash", ""))
    if _placeholder(stored): raise ValueError("Reward scale artifact hash is a placeholder")
    actual = canonical_payload_hash(payload)
    if stored != actual: raise ValueError("Reward scale artifact hash does not match contents")
    if expected_hash is not None:
        if _placeholder(expected_hash): raise ValueError("Expected reward scale hash is a placeholder")
        if str(expected_hash) != stored: raise ValueError("Expected reward scale hash does not match artifact")
    classification = str(payload.get("run_classification", "")).lower()
    if formal_mode and classification != "formal":
        raise ValueError("Formal mode rejects non-formal reward scale artifacts")
    if payload.get("validation_status") != "passed":
        raise ValueError("Reward scale artifact validation_status is not passed")
    bank_hash = payload.get("training_scenario_bank_hash")
    if expected_training_bank_hash is not None:
        if _placeholder(expected_training_bank_hash): raise ValueError("Expected training-bank hash is a placeholder")
        if bank_hash != expected_training_bank_hash: raise ValueError("Reward scale training-bank hash mismatch")
    scales = payload.get("scales", payload.get("raw_component_reference_scales"))
    if not isinstance(scales, dict): raise ValueError("Reward scale artifact missing scales")
    required = tuple(required_components)
    if tuple(payload.get("component_order", required)) != required:
        raise ValueError("Reward scale artifact component order mismatch")
    missing = [c for c in required if c not in scales]
    if missing: raise ValueError(f"Reward scale artifact missing required components: {missing}")
    components = payload.get("components", {})
    out = {}
    for k in required:
        meta = components.get(k, {}) if isinstance(components, dict) else {}
        if meta.get("status") in {"missing", "unexercised"}:
            raise ValueError(f"Reward scale component {k} has blocking status {meta.get('status')}")
        v = scales[k]
        if isinstance(v, bool): raise ValueError(f"Invalid boolean reward scale for {k}")
        f = float(v)
        if not math.isfinite(f) or f <= 0: raise ValueError(f"Reward scale for {k} must be finite and > 0")
        out[k] = f
    return RewardScaleArtifact(version, stored, out, p, classification, bank_hash, payload.get("estimator"))
