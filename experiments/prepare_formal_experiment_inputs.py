"""Prepare frozen formal scenario banks, reward scales, and resolved runtime configs."""
from __future__ import annotations
import argparse, copy, json, math, re
from pathlib import Path
from typing import Any
import yaml
from experiments.build_scenario_bank import build_bank
from experiments.estimate_reward_reference_scales import run_estimation
from evaluation.scenario_bank import load_bank_manifest, validate_disjoint_banks, verify_scenario_hashes, load_scenario_bank, sha256_file
from envs.reward_components import REWARD_COMPONENTS

PLACEHOLDER_RE = re.compile(r"REPLACE_WITH_(?:REAL|FINAL)_HASH|REPLACE_WITH_FINAL_[A-Z_]+_HASH|PLACEHOLDER|TBD")

def _freeze_value(cfg: dict[str, Any], dotted: str, default: int) -> int:
    cur: Any = cfg
    for part in dotted.split('.'):
        cur = cur.get(part, {}) if isinstance(cur, dict) else {}
    return int(cur.get('value', default)) if isinstance(cur, dict) else default

def _namespace(cfg: dict[str, Any]) -> int:
    return _freeze_value(cfg, 'seed_protocol.namespaces.scenario_generation', 10000)

def _validate_banks(paths: dict[str, Path], counts: dict[str, int]) -> dict[str, dict[str, Any]]:
    manifests = {split: load_bank_manifest(path) for split, path in paths.items()}
    for split, manifest in manifests.items():
        if manifest.get('split') != split or int(manifest.get('scenario_count', -1)) != counts[split]:
            raise ValueError(f"invalid {split} bank count/split")
        if not manifest.get('bank_hash') or str(manifest['bank_hash']).startswith('REPLACE'):
            raise ValueError(f"invalid {split} bank hash")
        for scenario in load_scenario_bank(paths[split]).scenarios:
            verify_scenario_hashes(scenario)
    validate_disjoint_banks(list(manifests.values()))
    return manifests

def _validate_reward_scale(path: Path, train_hash: str) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    if artifact.get('training_scenario_bank_hash') != train_hash:
        raise ValueError('reward scale train-bank lineage mismatch')
    if not artifact.get('artifact_hash') or str(artifact['artifact_hash']).startswith('REPLACE'):
        raise ValueError('invalid reward-scale artifact hash')
    missing = []
    for c in REWARD_COMPONENTS:
        rec = artifact.get('components', {}).get(c)
        scale = None if not rec else rec.get('scale')
        if rec is None or scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
            missing.append(c)
        if rec and rec.get('status') == 'instrumented_zero' and not rec.get('minimum_override'):
            raise ValueError(f'instrumented-zero override missing for {c}')
    if missing:
        raise ValueError('missing/nonpositive reward scales: ' + ', '.join(missing))
    return artifact

def _resolve_config(src: Path, dst: Path, manifests: dict[str, dict[str, Any]], scale: dict[str, Any], output_root: Path, reward_models: dict[str, dict[str, str]]) -> Path:
    cfg = yaml.safe_load(src.read_text()) or {}
    env = cfg.setdefault('env', {})
    env.update({
        'scenario_bank_manifest': str(output_root/'scenarios/train/scenario_bank_manifest.json'),
        'expected_bank_hash': manifests['train']['bank_hash'],
        'train_scenario_bank_manifest': str(output_root/'scenarios/train/scenario_bank_manifest.json'),
        'expected_train_bank_hash': manifests['train']['bank_hash'],
        'validation_scenario_bank_manifest': str(output_root/'scenarios/validation/scenario_bank_manifest.json'),
        'expected_validation_bank_hash': manifests['validation']['bank_hash'],
        'test_scenario_bank_manifest': str(output_root/'scenarios/test/scenario_bank_manifest.json'),
        'expected_test_bank_hash': manifests['test']['bank_hash'],
        'fallback': False,
    })
    if 'scenario_bank' in cfg:
        cfg['scenario_bank']['manifest'] = str(output_root/'scenarios/test/scenario_bank_manifest.json')
        cfg['scenario_bank']['expected_bank_hash'] = manifests['test']['bank_hash']
        cfg['scenario_bank']['expected_count'] = manifests['test']['scenario_count']
    if 'reward' in cfg:
        cfg['reward']['scale_artifact'] = str(output_root/'reward_scales/final_reward_reference_scales.json')
        cfg['reward']['scale_artifact_hash'] = scale['artifact_hash']
        cfg['reward']['expected_training_scenario_bank_hash'] = manifests['train']['bank_hash']
    if cfg.get('algorithm') == 'assignment_ppo':
        pass
    if cfg.get('rlaif', {}).get('agents'):
        for agent, rec in cfg['rlaif']['agents'].items():
            if rec.get('enabled') and agent in reward_models:
                rec['checkpoint'] = reward_models[agent]['path']; rec['checkpoint_hash'] = reward_models[agent]['hash']
    cfg.setdefault('output', {})['output_root'] = str(output_root/'runs'/src.stem)
    text = yaml.safe_dump(cfg, sort_keys=False)
    if PLACEHOLDER_RE.search(text):
        # unresolved reward-model hashes are allowed only in disabled/missing source templates, never runtime configs
        if 'checkpoint_hash: REPLACE_WITH_FINAL' in text:
            for agent, rec in cfg.get('rlaif', {}).get('agents', {}).items():
                if rec.get('enabled') and agent not in reward_models:
                    rec['checkpoint'] = None; rec['checkpoint_hash'] = None; rec['formal_checkpoint_status'] = 'missing_validated_formal_checkpoint'
            text = yaml.safe_dump(cfg, sort_keys=False)
        if PLACEHOLDER_RE.search(text):
            raise ValueError(f'unresolved placeholder in {src}')
    dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(text)
    return dst


def _write_resolved_configs(output_root: Path, manifests: dict[str, dict[str, Any]], scale_path: Path, scale_hash: str, reward_models: dict[str, dict[str, str]]) -> dict[str, str]:
    """Compatibility helper for focused tests: write runtime configs with real hashes."""
    output_root = Path(output_root)
    scale = {"artifact_hash": scale_hash, "training_scenario_bank_hash": manifests["train"]["bank_hash"]}
    resolved = {}
    for name in ("train_mappo_env", "train_mappo_rlaif_assignment", "train_mappo_rlaif_all", "train_assignment_ppo", "benchmark"):
        src = Path("configs/paper") / (name + ".yaml")
        if src.exists():
            resolved[name] = str(_resolve_config(src, output_root / "configs" / (name + ".resolved.yaml"), manifests, scale, output_root, reward_models))
    return resolved

def prepare(output_root: Path, *, force: bool=False, counts: dict[str,int]|None=None, scale_scenario_limit: int|None=None) -> dict[str, Any]:
    output_root = Path(output_root); counts = counts or {'train':300,'validation':60,'test':100}
    freeze = yaml.safe_load(Path('configs/paper/final_experiment_freeze.template.yaml').read_text()) or {}
    base = 'configs/paper/base_medium.yaml'; ns = _namespace(freeze)
    starts = {'train': ns, 'validation': ns + 100000, 'test': ns + 200000}
    paths = {s: output_root/'scenarios'/s for s in counts}
    for split in ('train','validation','test'):
        build_bank(base, split, counts[split], starts[split], paths[split], fallback=False, run_classification='formal', force=force)
    manifests = _validate_banks(paths, counts)
    scale_path = output_root/'reward_scales/final_reward_reference_scales.json'
    scale_cfg = copy.deepcopy(yaml.safe_load(Path('configs/paper/reward_scale_estimation.yaml').read_text()) or {})
    scale_cfg.setdefault('scenario_bank', {})['expected_bank_hash'] = manifests['train']['bank_hash']
    runtime_scale_cfg = output_root/'configs/reward_scale_estimation.resolved.yaml'; runtime_scale_cfg.parent.mkdir(parents=True, exist_ok=True)
    runtime_scale_cfg.write_text(yaml.safe_dump(scale_cfg, sort_keys=False))
    scale = run_estimation(paths['train'], runtime_scale_cfg, scale_path, run_classification='formal', scenario_limit=scale_scenario_limit, force=force)
    scale = _validate_reward_scale(scale_path, manifests['train']['bank_hash']) if scale is None else scale
    reward_models = {}
    for agent in ('assignment','truck','bus','station'):
        p = output_root/f'reward_models/reward_{agent}.pt'
        if p.is_file(): reward_models[agent] = {'path': str(p), 'hash': sha256_file(p)}
    resolved = {}
    for name in ('train_mappo_env','train_mappo_rlaif_assignment','train_mappo_rlaif_all','train_assignment_ppo','benchmark'):
        src = Path('configs/paper')/(name + '.yaml')
        if src.exists(): resolved[name] = str(_resolve_config(src, output_root/'configs'/(name+'.resolved.yaml'), manifests, scale, output_root, reward_models))
    manifest = {'scenario_banks': {k:{'path':str(paths[k]/'scenario_bank_manifest.json'),'count':manifests[k]['scenario_count'],'bank_hash':manifests[k]['bank_hash']} for k in manifests}, 'reward_scale': {'path': str(scale_path), 'artifact_hash': scale['artifact_hash'], 'training_scenario_bank_hash': scale['training_scenario_bank_hash'], 'estimator': scale.get('estimator'), 'component_validation_status': {k:v.get('status') for k,v in scale.get('components',{}).items()}}, 'resolved_configs': resolved, 'formal_reward_models': reward_models, 'rlaif_status': 'available' if 'assignment' in reward_models else 'RLAIF_BLOCKED_MISSING_FORMAL_ASSIGNMENT_REWARD_CHECKPOINT'}
    (output_root/'formal_input_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root', type=Path, default=Path('results/formal')); ap.add_argument('--force', action='store_true'); ap.add_argument('--train-count', type=int, default=300); ap.add_argument('--validation-count', type=int, default=60); ap.add_argument('--test-count', type=int, default=100); ap.add_argument('--scale-scenario-limit', type=int)
    a=ap.parse_args(argv)
    m=prepare(a.output_root, force=a.force, counts={'train':a.train_count,'validation':a.validation_count,'test':a.test_count}, scale_scenario_limit=a.scale_scenario_limit)
    print(json.dumps(m, indent=2, sort_keys=True)); return 0
if __name__ == '__main__': raise SystemExit(main())
