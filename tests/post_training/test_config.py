import pytest
from training.post_training.config import PostTrainingConfigError, resolve_post_training_config

def base():
    return {"method_id":"mappo_env_post_base", "training_stage":"base_pretraining",
            "training":{"seed":1,"initialization":{"mode":"random"}}}

def test_random_base_initialization_config_succeeds():
    assert resolve_post_training_config(base())["algorithm"].endswith("post_base")

@pytest.mark.parametrize("field,value", [("load_optimizers", True), ("strict", False)])
def test_unsafe_loading_is_rejected(field, value):
    config=base(); config["training"]["initialization"][field]=value
    with pytest.raises(PostTrainingConfigError): resolve_post_training_config(config)

def test_child_cannot_fall_back_to_random():
    config=base(); config.update(method_id="mappo_env_post_continued", training_stage="environment_post_training")
    with pytest.raises(PostTrainingConfigError, match="pretrained_policy"): resolve_post_training_config(config)

def test_missing_reward_model_fails_closed(tmp_path):
    config=base(); config.update(method_id="mappo_rlaif_assignment_post", training_stage="rlaif_post_training")
    config["training"]["initialization"].update(mode="pretrained_policy", checkpoint_path=str(tmp_path/"x"), manifest_path=str(tmp_path/"y"))
    config["rlaif"]={"scope":"assignment", "reward_model_path":str(tmp_path/"missing")}
    with pytest.raises(PostTrainingConfigError): resolve_post_training_config(config)
