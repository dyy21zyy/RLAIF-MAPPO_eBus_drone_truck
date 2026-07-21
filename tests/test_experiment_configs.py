from evaluation.result_schema import RESULT_FIELDS
from utils.config import load_config

def test_experiment_configs_have_required_sections():
    benchmark=load_config("configs/experiments.yaml"); assert benchmark["experiment"]["seeds"] and len(benchmark["methods"])>=9
    ablation=load_config("configs/ablation.yaml"); assert {x["name"] for x in ablation["ablations"]}>={"no_rlaif","with_rlaif","no_action_mask","no_bus_learning","assignment_ppo_only","mappo_async","delayed_reward_off","delayed_reward_on"}
    sensitivity=load_config("configs/sensitivity.yaml"); assert len(sensitivity["dimensions"])==9

def test_result_schema_required_fields():
    required={"experiment_id","method_name","seed","total_env_reward","total_rlaif_reward","fallback_feasibility_events","status","error_message"}; assert required<=set(RESULT_FIELDS)

def test_stage8_smoke_reports_runtime_torch_availability(monkeypatch, capsys):
    import experiments.smoke_test_experiments as smoke
    monkeypatch.setattr(smoke, "is_torch_runtime_available", lambda: False)
    assert smoke.main() == 0
    assert "torch_available=False" in capsys.readouterr().out
