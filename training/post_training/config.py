"""Fail-closed configuration for the isolated policy post-training pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

METHOD_ALGORITHMS = {
    "mappo_env_post_base": "four_agent_asynchronous_mappo_env_post_base",
    "mappo_env_post_continued": "four_agent_asynchronous_mappo_env_post_continued",
    "mappo_rlaif_assignment_post": "four_agent_asynchronous_mappo_rlaif_assignment_post",
}


class PostTrainingConfigError(ValueError):
    """A post-training contract is incomplete or unsafe."""


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise PostTrainingConfigError(f"missing required post-training field: {key}")
    return mapping[key]


def resolve_post_training_config(source: str | Path | dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
    """Load and validate an isolated post-training configuration.

    Paths are checked here so a bad pretrained initialization can never silently
    become random initialization.
    """
    if isinstance(source, (str, Path)):
        config = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    else:
        config = deepcopy(source)
    method = str(_require(config, "method_id"))
    if method not in METHOD_ALGORITHMS:
        raise PostTrainingConfigError(f"unsupported post-training method_id: {method}")
    config["algorithm"] = METHOD_ALGORITHMS[method]
    expected_stage = "base_pretraining" if method == "mappo_env_post_base" else (
        "rlaif_post_training" if method == "mappo_rlaif_assignment_post" else "environment_post_training"
    )
    if config.get("training_stage") != expected_stage:
        raise PostTrainingConfigError(f"{method} requires training_stage={expected_stage}")
    training = _require(config, "training")
    if seed is not None:
        training["seed"] = int(seed)
    training["seed"] = int(_require(training, "seed"))
    initialization = _require(training, "initialization")
    required_mode = "random" if method == "mappo_env_post_base" else "pretrained_policy"
    if initialization.get("mode") != required_mode:
        raise PostTrainingConfigError(f"{method} requires initialization.mode={required_mode}")
    if initialization.get("load_optimizers", False):
        raise PostTrainingConfigError("post-training never loads parent optimizers")
    if initialization.get("strict", True) is not True:
        raise PostTrainingConfigError("post-training initialization must be strict")
    initialization.setdefault("load_actors", required_mode == "pretrained_policy")
    initialization.setdefault("load_critic", required_mode == "pretrained_policy")
    initialization.setdefault("load_optimizers", False)
    initialization.setdefault("strict", True)
    if required_mode == "pretrained_policy":
        if not initialization["load_actors"] or not initialization["load_critic"]:
            raise PostTrainingConfigError("pretrained_policy must load all actors and the critic")
        for key in ("checkpoint_path", "manifest_path"):
            path = Path(str(_require(initialization, key))).expanduser().resolve()
            if not path.is_file():
                raise PostTrainingConfigError(f"{key} does not exist: {path}")
            initialization[key] = str(path)
        initialization.setdefault("expected_parent_method_id", "mappo_env_post_base")
        initialization.setdefault("expected_parent_algorithm", "four_agent_asynchronous_mappo_env_post_base")
        initialization.setdefault("expected_parent_rlaif_scope", "none")
    rlaif = config.setdefault("rlaif", {})
    if method == "mappo_rlaif_assignment_post":
        if rlaif.get("scope") != "assignment":
            raise PostTrainingConfigError("RLAIF post-training requires rlaif.scope=assignment")
        if rlaif.get("fallback_to_env_reward", False) is not False:
            raise PostTrainingConfigError("RLAIF post-training forbids reward fallback")
        if rlaif.get("fail_on_invalid_reward_model", True) is not True:
            raise PostTrainingConfigError("RLAIF post-training must fail on an invalid reward model")
        reward_path = Path(str(rlaif.get("reward_model_path", ""))).expanduser()
        if not reward_path.is_file():
            raise PostTrainingConfigError(f"assignment Reward Model does not exist: {reward_path}")
    return config
