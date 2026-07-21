from training.mappo_trainer import CHECKPOINT_SCHEMA_VERSION, _require_checkpoint_compatible

def good():
    return {"checkpoint_schema_version":CHECKPOINT_SCHEMA_VERSION,"event_schema_version":1,"observation_schema_version":3,"candidate_schema_version":2,"entity_encoder_schema_version":1,"stage":7,"algorithm":"four_agent_asynchronous_mappo_env_reward_only"}

def test_checkpoint_stores_schema_versions_and_rejects_incompatible():
    _require_checkpoint_compatible(good())
    bad = good(); bad["candidate_schema_version"] = -1
    try:
        _require_checkpoint_compatible(bad)
    except ValueError as exc:
        assert "candidate_schema_version" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint accepted")
