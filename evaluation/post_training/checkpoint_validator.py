"""File hash and manifest validation for isolated checkpoints."""
import json
from pathlib import Path
from experiments.post_training.artifacts import sha256_file
from .policy_registry import validate_policy
def validate_checkpoint(path:Path,manifest_path:Path):
    if not path.is_file() or not manifest_path.is_file(): raise FileNotFoundError(path if not path.is_file() else manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("checkpoint_sha256")!=sha256_file(path): raise ValueError("checkpoint hash mismatch")
    validate_policy(manifest); return manifest
