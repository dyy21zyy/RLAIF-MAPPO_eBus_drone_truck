from experiments.train_all_reward_models import build_reward_model_matrix_manifest
def test_all_four_ready_false_when_one_fails():
 d={a:{'validation_status':'passed','run_classification':'formal'} for a in ('assignment','truck','bus','station')}; d['bus']['validation_status']='failed_test_accuracy'; assert not build_reward_model_matrix_manifest(d)['all_four_formal_models_ready']
def test_all_four_ready_true_only_when_all_pass():
 d={a:{'validation_status':'passed','run_classification':'formal'} for a in ('assignment','truck','bus','station')}; assert build_reward_model_matrix_manifest(d)['all_four_formal_models_ready']
