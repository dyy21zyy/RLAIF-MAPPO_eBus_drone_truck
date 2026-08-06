"""Episode row schema helper; execution remains explicit in benchmark CLI."""
REQUIRED=("method_id","algorithm","training_stage","training_seed","scenario_id","scenario_content_hash","policy_checkpoint_path","policy_checkpoint_sha256","parent_checkpoint_path","parent_checkpoint_sha256","reward_checkpoint_paths","reward_checkpoint_hashes","reward_scale_artifact_path","reward_scale_artifact_hash","code_commit","formal_metrics","status")
def validate_episode_row(row):
    missing=[k for k in REQUIRED if k not in row]
    if missing: raise ValueError(f"benchmark row missing {missing}")
    return row
