from experiments.validate_formal_experiment_readiness import validate_readiness, CONFIG_VALID_ARTIFACTS_MISSING, CONFIG_INVALID
import yaml
def test_missing_bank(tmp_path):
    c=tmp_path/'c.yaml'; c.write_text(yaml.safe_dump({'run_classification':'formal','fallback':False,'paired_evaluation':True,'scenario_bank':{'manifest':str(tmp_path/'missing.json'),'expected_count':1},'methods':[{'method_id':'truck_direct_heuristic'}]}))
    assert validate_readiness(c)['status']==CONFIG_VALID_ARTIFACTS_MISSING
def test_fallback_blocks(tmp_path):
    c=tmp_path/'c.yaml'; c.write_text(yaml.safe_dump({'run_classification':'formal','fallback':True,'scenario_bank':{'manifest':str(tmp_path/'missing.json')},'methods':[]}))
    assert validate_readiness(c)['status']==CONFIG_INVALID


def test_multiseed_policy_checkpoints_are_validated(tmp_path, monkeypatch):
    import experiments.validate_formal_experiment_readiness as readiness

    bank = tmp_path / "scenario_bank_manifest.json"
    bank.write_text("{}", encoding="utf-8")

    checkpoints = {}
    expected_calls = set()

    for method_id in ("mappo_env", "mappo_rlaif_assignment"):
        checkpoints[method_id] = {}
        for seed in (1, 2, 3):
            path = tmp_path / f"{method_id}_seed_{seed}.pt"
            path.write_bytes(b"checkpoint")
            checkpoints[method_id][str(seed)] = str(path)
            expected_calls.add((method_id, seed, str(path)))

    reward = tmp_path / "reward_assignment.pt"
    reward.write_bytes(b"reward")

    monkeypatch.setattr(
        readiness,
        "load_bank_manifest",
        lambda _path: {"scenario_count": 1, "split": "test"},
    )

    calls = set()

    def fake_validate(spec, path):
        calls.add((spec.method_id, spec.training_seed, str(path)))
        return {}

    monkeypatch.setattr(readiness, "validate_policy_checkpoint", fake_validate)
    monkeypatch.setattr(
        readiness,
        "load_strict_agent_reward_checkpoint",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        readiness,
        "validate_unique_learned_checkpoints",
        lambda specs: None,
    )

    config = {
        "run_classification": "formal",
        "fallback": False,
        "paired_evaluation": True,
        "scenario_bank": {
            "manifest": str(bank),
            "split": "test",
            "expected_count": 1,
        },
        "training_seeds": [1, 2, 3],
        "methods": [
            {
                "method_id": "mappo_env",
                "policy_checkpoints": checkpoints["mappo_env"],
            },
            {
                "method_id": "mappo_rlaif_assignment",
                "policy_checkpoints": checkpoints["mappo_rlaif_assignment"],
                "reward_checkpoints": {
                    "assignment": str(reward),
                },
            },
        ],
    }

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = readiness.validate_readiness(config_path)

    assert result["overall_status"] == readiness.READY_FOR_FORMAL_EVALUATION
    assert result["missing_artifacts"] == []
    assert calls == expected_calls
