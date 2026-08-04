from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_assignment_preference_target_supports_validation_minima() -> None:
    preference_config = yaml.safe_load(
        (
            ROOT
            / "configs/paper/rlaif_preference_generation.yaml"
        ).read_text()
    )
    reward_config = yaml.safe_load(
        (
            ROOT
            / "configs/paper/train_reward_assignment.yaml"
        ).read_text()
    )

    target = int(
        preference_config["agents"]["assignment"][
            "target_valid_pair_count"
        ]
    )
    maximum_attempts = int(
        preference_config["agents"]["assignment"][
            "max_api_attempts"
        ]
    )

    split = reward_config["split"]
    validation = reward_config["validation"]

    minimum_required = max(
        math.ceil(
            validation["minimum_training_pairs"]
            / split["train_fraction"]
        ),
        math.ceil(
            validation["minimum_validation_pairs"]
            / split["validation_fraction"]
        ),
        math.ceil(
            validation["minimum_test_pairs"]
            / split["test_fraction"]
        ),
    )

    assert target >= minimum_required, (
        f"assignment target {target} cannot satisfy the reward-model "
        f"split minima; at least {minimum_required} pairs are required"
    )
    assert maximum_attempts >= target
