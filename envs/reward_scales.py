from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math

PLACEHOLDERS = {"freeze-after-estimation", "tbd", "unknown", "placeholder", ""}

@dataclass(frozen=True)
class RewardScaleArtifact:
    artifact_version: int
    artifact_hash: str
    scales: dict[str, float]
    source_path: Path

def _canonical_payload_hash(payload: dict) -> str:
    clean = dict(payload); clean.pop("artifact_hash", None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()

def load_reward_scale_artifact(path: str | Path, *, expected_hash: str | None, required_components: set[str]) -> RewardScaleArtifact:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reward scale artifact not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reward scale JSON: {p}") from exc
    version = int(payload.get("artifact_version", -1))
    if version != 1:
        raise ValueError(f"Unsupported reward scale artifact_version: {version}")
    stored = str(payload.get("artifact_hash", ""))
    if stored.lower() in PLACEHOLDERS:
        raise ValueError("Reward scale artifact hash is a placeholder")
    actual = _canonical_payload_hash(payload)
    if stored != actual:
        raise ValueError("Reward scale artifact hash does not match contents")
    if expected_hash is not None:
        if str(expected_hash).lower() in PLACEHOLDERS:
            raise ValueError("Expected reward scale hash is a placeholder")
        if str(expected_hash) != stored:
            raise ValueError("Expected reward scale hash does not match artifact")
    scales = payload.get("scales", payload.get("raw_component_reference_scales"))
    if not isinstance(scales, dict):
        raise ValueError("Reward scale artifact missing scales")
    missing = sorted(required_components - set(scales))
    if missing:
        raise ValueError(f"Reward scale artifact missing required components: {missing}")
    out = {}
    for k, v in scales.items():
        if isinstance(v, bool):
            raise ValueError(f"Invalid boolean reward scale for {k}")
        f = float(v)
        if not math.isfinite(f) or f <= 0:
            raise ValueError(f"Reward scale for {k} must be finite and > 0")
        out[str(k)] = f
    return RewardScaleArtifact(version, stored, out, p)
