from evaluation.preformal_part3_gates import method_matrix

def test_method_matrix_scopes_and_missing_checkpoint_blocks_only_learned():
    rows={r['method_id']:r for r in method_matrix({'assignment_ppo':'a.json','mappo_env':'e.json','mappo_rlaif_assignment':'ra.json'})}
    assert rows['truck_direct_heuristic']['enabled'] and rows['truck_direct_heuristic']['policy_checkpoint'] is None
    assert rows['assignment_ppo']['policy_checkpoint']=='a.json' and rows['assignment_ppo']['enabled_reward_agents']==[]
    assert rows['mappo_env']['rlaif_scope']=='none' and rows['mappo_env']['enabled_reward_agents']==[]
    assert rows['mappo_rlaif_assignment']['enabled_reward_agents']==['assignment']
    assert rows['mappo_rlaif_all']['status']=='blocked_missing_artifact'
