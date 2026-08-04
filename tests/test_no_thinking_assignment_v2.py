from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest
import yaml

from experiments.generate_formal_multiagent_preferences import (
    build_prompt,
    evaluator_settings,
)
from rlaif.ai_evaluator import APISettings, _default_api_call
from rlaif.preference_quality import validate_assignment_v2


class DummyHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, *args, **kwargs) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def candidate(
    candidate_id: str,
    delivery: float,
    lateness: float,
    distance: float,
    truck_time: float,
) -> dict:
    names = [
        "estimated_delivery_time_norm",
        "estimated_lateness_norm",
        "estimated_truck_distance_norm",
        "estimated_truck_time_norm",
    ]
    return {
        "candidate_id": candidate_id,
        "action_id": 0,
        "feature_names": names,
        "features": [
            delivery,
            lateness,
            distance,
            truck_time,
        ],
        "feasible": True,
    }


def supported_response(criteria: list[str]) -> dict:
    return {
        "preferred": "A",
        "confidence": 0.9,
        "criteria": criteria,
        "evidence": [
            {
                "metric": "estimated_delivery_time_norm",
                "better_candidate": "A",
            },
            {
                "metric": "estimated_lateness_norm",
                "better_candidate": "A",
            },
            {
                "metric": "estimated_truck_distance_norm",
                "better_candidate": "A",
            },
            {
                "metric": "estimated_truck_time_norm",
                "better_candidate": "A",
            },
        ],
        "reason": (
            "Candidate A is preferred because it has lower delivery time, "
            "expected lateness, truck distance, and truck time."
        ),
    }


def test_default_api_call_disables_thinking_and_uses_configured_timeout(
    monkeypatch,
):
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return DummyHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"preferred":"A"}',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    settings = APISettings(
        api_key="secret",
        api_base_url="https://example.test/v1",
        model_name="qwen3.7-plus",
        temperature=0.0,
        max_retries=3,
        enable_thinking=False,
        timeout_seconds=12.5,
    )

    _default_api_call("test prompt", settings)

    body = json.loads(captured["request"].data.decode("utf-8"))

    assert body["enable_thinking"] is False
    assert body["model"] == "qwen3.7-plus"
    assert captured["timeout"] == 12.5


def test_formal_evaluator_settings_require_thinking_disabled(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv("OPENAI_MODEL", "qwen3.7-plus")

    cfg = {
        "evaluator": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "model_env": "OPENAI_MODEL",
            "temperature": 0.0,
            "max_retries": 3,
            "enable_thinking": False,
            "timeout_seconds": 17.5,
        }
    }

    settings = evaluator_settings(cfg)

    assert settings.enable_thinking is False
    assert settings.timeout_seconds == 17.5

    cfg["evaluator"]["enable_thinking"] = True

    with pytest.raises(
        ValueError,
        match="enable_thinking=false",
    ):
        evaluator_settings(cfg)


@pytest.mark.parametrize(
    "criteria",
    [
        [
            "delivery_time",
            "lateness",
            "truck_distance",
            "truck_time",
        ],
        [
            "estimated_delivery_time_norm",
            "estimated_lateness_norm",
            "estimated_truck_distance_norm",
            "estimated_truck_time_norm",
        ],
    ],
)
def test_assignment_v2_safely_canonicalizes_supported_aliases(
    criteria,
):
    a = candidate("TD_0", 0.20, 0.05, 0.70, 0.10)
    b = candidate("TLD_1", 0.30, 0.10, 0.90, 0.20)

    validated = validate_assignment_v2(
        supported_response(criteria),
        a,
        b,
    )

    assert validated["criteria"] == [
        "delivery_time",
        "expected_lateness",
        "truck_distance",
        "truck_time",
    ]

    assert validated["validated_evidence"] == [
        {
            "metric": "delivery_time",
            "better_candidate": "A",
        },
        {
            "metric": "expected_lateness",
            "better_candidate": "A",
        },
        {
            "metric": "truck_distance",
            "better_candidate": "A",
        },
        {
            "metric": "truck_time",
            "better_candidate": "A",
        },
    ]


def test_assignment_v2_still_rejects_energy_criterion():
    a = candidate("TD_0", 0.20, 0.05, 0.70, 0.10)
    b = candidate("TLD_1", 0.30, 0.10, 0.90, 0.20)

    response = supported_response(["energy"])

    with pytest.raises(
        ValueError,
        match="illegal or ambiguous criterion",
    ):
        validate_assignment_v2(response, a, b)


def test_assignment_prompt_lists_exact_canonical_vocabulary():
    config = yaml.safe_load(
        Path(
            "configs/paper/rlaif_preference_generation.yaml"
        ).read_text(encoding="utf-8")
    )

    a = candidate("TD_0", 0.20, 0.05, 0.70, 0.10)
    b = candidate("TLD_1", 0.30, 0.10, 0.90, 0.20)

    state = {
        "agent_type": "assignment",
        "event_type": "PARCEL_RELEASE",
        "scenario_id": "scenario-test",
        "simulation_time": 0.0,
        "state_feature_names": ["priority_urgent"],
        "state_features": [1.0],
    }

    prompt = build_prompt(state, a, b, config)

    canonical = [
        "delivery_feasibility",
        "delivery_time",
        "expected_lateness",
        "deadline_risk",
        "truck_distance",
        "truck_time",
        "truck_capacity",
        "bus_wait_time",
        "bus_linehaul_time",
        "bus_freight_capacity",
        "drone_time",
        "drone_feasibility",
        "locker_congestion",
        "station_power_margin",
        "downstream_congestion",
    ]

    for criterion in canonical:
        assert criterion in prompt

    assert "TLD does not use a bus" in prompt
    assert "TD does not use a locker or drone" in prompt
    assert "feature names must not be copied into criteria" in prompt


def test_formal_config_explicitly_disables_thinking():
    config = yaml.safe_load(
        Path(
            "configs/paper/rlaif_preference_generation.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["evaluator"]["enable_thinking"] is False
    assert config["evaluator"]["timeout_seconds"] == 60
