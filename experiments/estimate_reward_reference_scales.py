"""Estimate robust reward reference scales for Phase 7 environment-only MAPPO."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from training.ppo_trainer import create_environment
from utils.config import load_config

ARTIFACT_VERSION = 1
REFERENCE_WEIGHTS = {
    "passenger_delay": 1.0,
    "parcel_lateness": 1.0,
    "energy_cost": 0.2,
    "power_overload": 1.0,
    "bus_battery_violation": 5.0,
    "locker_overflow": 1.0,
}

def _first_feasible(obs):
    return next(i for i, ok in enumerate(obs["action_mask"]) if ok)

def estimate(config_path: str, episodes: int = 3, output: str | None = None):
    loaded = load_config(config_path)
    config = loaded if "env" in loaded else {"env": {"config_path": str(config_path), "fallback": True}, "project": loaded.get("project", {}), "reward": loaded.get("reward", {})}
    env = create_environment(config)
    raw = {k: [] for k in set(REFERENCE_WEIGHTS) | set(config.get("reward", {}))}
    for seed in range(episodes):
        obs, _ = env.reset(seed=int(config.get("project",{}).get("seed",0))+seed)
        while obs["agent_id"] != "terminal":
            obs, _reward, term, trunc, info = env.step(_first_feasible(obs))
            for name, value in info.get("cost_components", {}).items():
                weight = float(config.get("reward", {}).get(name, 1.0))
                if weight:
                    raw.setdefault(name, []).append(abs(float(value)) / abs(weight))
            if term or trunc:
                break
    scales = {}
    for name, values in raw.items():
        nz = np.asarray([v for v in values if v > 0], dtype=float)
        scales[name] = float(np.percentile(nz, 90)) if len(nz) else 1.0
    payload = {"artifact_version": ARTIFACT_VERSION, "config": str(config_path), "episodes": episodes, "reference_weights": REFERENCE_WEIGHTS, "raw_component_reference_scales": scales, "frozen": True}
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["artifact_hash"] = hashlib.sha256(encoded).hexdigest()
    path = Path(output or "outputs/reward_reference_scales_v1.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", default=None)
    args=ap.parse_args()
    print(json.dumps(estimate(args.config, args.episodes, args.output), indent=2, sort_keys=True))
if __name__ == "__main__": main()
