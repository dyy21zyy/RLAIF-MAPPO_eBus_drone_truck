"""Semantic guards preventing the isolated workflow from changing the baseline."""
from pathlib import Path
import ast
import yaml

from evaluation.formal_policy_registry import FORMAL_METHOD_REGISTRY
from experiments.post_training import METHODS, OUTPUT_ROOT
from experiments.run_hard_locker_post_training_experiment import PHASES as POST_PHASES
from training.post_training.checkpoint import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]


def test_original_configs_keep_reward_and_random_initialization_semantics():
    env = yaml.safe_load((ROOT / "configs/paper/train_mappo_env.yaml").read_text())
    aligned = yaml.safe_load((ROOT / "configs/paper/train_mappo_rlaif_assignment.yaml").read_text())
    assert env["mode"] == "environment_reward"
    assert not env["rlaif"]["enabled"] and env["output"]["output_root"] == "results/formal/mappo_env"
    assert aligned["mode"] == "rlaif_reward"
    assert aligned["rlaif"]["enabled"] and aligned["rlaif"]["agents"]["assignment"]["enabled"]
    assert "initialization" not in aligned["training"]  # legacy construction is random


def test_pipeline_roots_phases_and_method_namespaces_are_isolated():
    legacy = (ROOT / "experiments/run_hard_locker_reexperiment.py").read_text()
    assert 'Path("results/formal")' in legacy and "choices=range(10)" in legacy
    assert len(POST_PHASES) == 12 and OUTPUT_ROOT == "results/formal_post_training"
    assert set(METHODS).isdisjoint(FORMAL_METHOD_REGISTRY)


def test_entrypoints_do_not_dispatch_to_each_other_or_overwrite_outputs():
    old_tree = ast.parse((ROOT / "experiments/train_mappo_async.py").read_text())
    old_imports = {node.module for node in ast.walk(old_tree) if isinstance(node, ast.ImportFrom)}
    assert not any(module and module.startswith("training.post_training") for module in old_imports)
    new_text = (ROOT / "experiments/train_mappo_post_training.py").read_text()
    assert "results/formal/" not in new_text


def test_checkpoint_schemas_are_independent():
    import training.mappo_trainer as legacy
    assert legacy.CHECKPOINT_SCHEMA_VERSION == 4
    assert SCHEMA_VERSION == 1
    assert "post_training_checkpoint_schema_version" not in legacy.save_checkpoint.__code__.co_consts
