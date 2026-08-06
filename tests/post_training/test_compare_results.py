import json
import pytest
from experiments.compare_original_and_post_training_results import compare, main


def _rows(methods, digest="hash"):
    return [{"scenario_id":"s1","scenario_content_hash":digest,"method_id":m,"episode_reward":v}
            for m,v in methods.items()]


def test_exploratory_comparisons_and_label():
    old=_rows({"mappo_env":1,"mappo_rlaif_assignment":2})
    post=_rows({"mappo_env_post_continued":3,"mappo_rlaif_assignment_post":5})
    result=compare(old,post,"episode_reward")
    assert result["analysis_type"] == "exploratory_cross_experiment_not_confirmatory"
    assert result["deltas"] == {"original_rlaif_vs_original_mappo_env":1,
      "rlaif_post_vs_environment_continuation":2,"rlaif_post_vs_original_rlaif":3}


def test_hash_mismatch_fails_closed():
    with pytest.raises(ValueError,match="IDs and hashes"):
        compare(_rows({"mappo_env":1,"mappo_rlaif_assignment":2}),
          _rows({"mappo_env_post_continued":3,"mappo_rlaif_assignment_post":4},"tampered"),"episode_reward")


def test_optional_artifacts_do_not_block_pipeline(capsys):
    assert main([])==0
    assert "not run" in capsys.readouterr().out
