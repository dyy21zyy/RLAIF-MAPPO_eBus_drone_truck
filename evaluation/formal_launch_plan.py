"""Formal launch-plan generator. Produces commands; never executes them."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

STAGES=(
 'scenario_bank_generation','scenario_bank_validation','reward_reference_scale_estimation','reward_scale_validation','reward_model_validation','assignment_ppo_training','environment_mappo_training','assignment_rlaif_mappo_training','full_rlaif_mappo_training','final_benchmark','final_ablation_matrix','final_sensitivity_matrix','final_aggregation','final_result_integrity_validation')
LEARNED=('assignment_ppo','mappo_env','mappo_rlaif_assignment','mappo_rlaif_all')
HEURISTICS=('truck_direct_heuristic','integrated_rule_based')

def _hash(x:Any)->str: return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def _ready_hash(h): return bool(h) and h not in {'REPLACE_WITH_REAL_HASH','PLACEHOLDER',''} and not str(h).startswith('0000')
def _cmd(parts): return ' '.join(str(p) for p in parts)

def generate_formal_launch_plan(resolved:dict, output_path:str|Path)->dict:
    seeds=resolved.get('training_seeds',[1,2,3]); episodes=int(resolved.get('mappo_total_episodes',3000)); test_scenarios=int(resolved.get('test_scenarios',100))
    hashes=resolved.get('input_hashes',{})
    stages=[]
    def add(stage, command, *, config_path, output_dir, expected_predecessor_artifact=None, expected_input_hash=None, training_seed=None, resume_behavior='resume_if_identity_hashes_match'):
        blocked=[]
        if expected_input_hash is not None and not _ready_hash(expected_input_hash): blocked.append('blocked_missing_artifact')
        if any(s in str(command) for s in ('REPLACE_WITH','diagnostic','smoke','{{','}}')): blocked.append('blocked_missing_artifact')
        stages.append({'stage_id':stage,'status':'blocked_missing_artifact' if blocked else 'ready','command':None if blocked else command,'blocked_reasons':blocked,'config_path':config_path,'training_seed':training_seed,'expected_predecessor_artifact':expected_predecessor_artifact,'expected_input_hash':expected_input_hash,'output_directory':output_dir,'resume_behavior':resume_behavior})
    add('scenario_bank_generation', _cmd(['python','-m','experiments.generate_scenario_bank','--config',resolved.get('scenario_generation_config','configs/paper/scenario_bank.yaml'),'--output-root','results/formal/scenario_banks','--test-count',test_scenarios]), config_path=resolved.get('scenario_generation_config','configs/paper/scenario_bank.yaml'), output_dir='results/formal/scenario_banks', expected_input_hash=hashes.get('scenario_generation_config_hash','required'))
    add('scenario_bank_validation', _cmd(['python','-m','experiments.validate_scenario_bank','--manifest','results/formal/scenario_banks/test_manifest.json','--expected-count',test_scenarios]), config_path='results/formal/scenario_banks/test_manifest.json', output_dir='results/formal/validation/scenario_bank', expected_predecessor_artifact='results/formal/scenario_banks/test_manifest.json', expected_input_hash=hashes.get('scenario_bank_hash'))
    add('reward_reference_scale_estimation', _cmd(['python','-m','experiments.estimate_reward_reference_scale','--config',resolved.get('reward_scale_config','configs/paper/reward_scale.yaml'),'--output-root','results/formal/reward_scale']), config_path=resolved.get('reward_scale_config','configs/paper/reward_scale.yaml'), output_dir='results/formal/reward_scale', expected_input_hash=hashes.get('scenario_bank_hash'))
    add('reward_scale_validation', _cmd(['python','-m','experiments.validate_reward_scale','--config','results/formal/reward_scale/reward_scale.json']), config_path='results/formal/reward_scale/reward_scale.json', output_dir='results/formal/validation/reward_scale', expected_input_hash=hashes.get('reward_scale_hash'))
    add('reward_model_validation', _cmd(['python','-m','experiments.validate_reward_models','--config',resolved.get('reward_model_validation_config','configs/paper/reward_models.yaml')]), config_path=resolved.get('reward_model_validation_config','configs/paper/reward_models.yaml'), output_dir='results/formal/validation/reward_models', expected_input_hash=hashes.get('reward_model_hash'))
    train_cfg={'assignment_ppo':'results/formal/configs/assignment_ppo.yaml','mappo_env':'results/formal/configs/mappo_env.yaml','mappo_rlaif_assignment':'results/formal/configs/mappo_rlaif_assignment.yaml','mappo_rlaif_all':'results/formal/configs/mappo_rlaif_all.yaml'}
    stage_map={'assignment_ppo':'assignment_ppo_training','mappo_env':'environment_mappo_training','mappo_rlaif_assignment':'assignment_rlaif_mappo_training','mappo_rlaif_all':'full_rlaif_mappo_training'}
    for method in LEARNED:
        for seed in seeds:
            out=f'results/formal/{method}/seed_{seed}'
            add(stage_map[method], _cmd(['python','-m','experiments.train_mappo_async' if method!='assignment_ppo' else 'experiments.train_assignment_ppo','--config',train_cfg[method],'--seed',seed,'--total-episodes',episodes,'--output-root',out]), config_path=train_cfg[method], training_seed=seed, output_dir=out, expected_input_hash=hashes.get('scenario_bank_hash'))
    add('final_benchmark', _cmd(['python','-m','experiments.run_paper_benchmark','--config','results/formal/configs/benchmark.yaml','--output-root','results/formal/benchmark']), config_path='results/formal/configs/benchmark.yaml', output_dir='results/formal/benchmark', expected_input_hash=hashes.get('benchmark_config_hash'))
    add('final_ablation_matrix', _cmd(['python','-m','experiments.run_ablation_matrix','--config','results/formal/configs/ablation_matrix.resolved.yaml','--output-root','results/formal/ablation']), config_path='results/formal/configs/ablation_matrix.resolved.yaml', output_dir='results/formal/ablation', expected_input_hash=hashes.get('ablation_config_hash'))
    add('final_sensitivity_matrix', _cmd(['python','-m','experiments.run_sensitivity_matrix','--config','results/formal/configs/sensitivity_matrix.resolved.yaml','--output-root','results/formal/sensitivity']), config_path='results/formal/configs/sensitivity_matrix.resolved.yaml', output_dir='results/formal/sensitivity', expected_input_hash=hashes.get('sensitivity_config_hash'))
    add('final_aggregation', _cmd(['python','-m','experiments.aggregate_paper_results','--config','results/formal/configs/aggregation.resolved.yaml','--output-root','results/formal/aggregation']), config_path='results/formal/configs/aggregation.resolved.yaml', output_dir='results/formal/aggregation', expected_input_hash=hashes.get('benchmark_config_hash'))
    add('final_result_integrity_validation', _cmd(['python','-m','experiments.validate_formal_experiment_readiness','--config','results/formal/configs/result_integrity.resolved.yaml']), config_path='results/formal/configs/result_integrity.resolved.yaml', output_dir='results/formal/validation/result_integrity', expected_input_hash=hashes.get('result_integrity_config_hash','required'))
    learned_rows=len(LEARNED)*len(seeds)*test_scenarios; heuristic_rows=len(HEURISTICS)*test_scenarios
    plan={'publication_eligible':False,'training_seeds':seeds,'mappo_total_episodes':episodes,'test_scenarios':test_scenarios,'learned_methods':list(LEARNED),'heuristic_methods':list(HEURISTICS),'expected_benchmark_rows':{'formula':'(learned_methods * training_seeds * test_scenarios) + (heuristic_methods * test_scenarios)','learned_rows':learned_rows,'heuristic_rows':heuristic_rows,'total':learned_rows+heuristic_rows},'stages':stages}
    plan['formal_launch_plan_hash']=_hash({k:v for k,v in plan.items() if k!='formal_launch_plan_hash'})
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); Path(output_path).write_text(json.dumps(plan,indent=2,sort_keys=True))
    return plan
