from evaluation.artifact_manifest import build_manifest, REQUIRED

def test_manifests_contain_required_hashes():
    m=build_manifest(scenario_manifest_hash='abc', preference_data_hash='def', reward_checkpoint_hashes={'a':'1'}, policy_checkpoint_hash='2')
    for k in REQUIRED: assert k in m
