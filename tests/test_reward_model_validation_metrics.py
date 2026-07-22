import math, torch
from rlaif.reward_model_trainer import determine_validation_status
def test_nonfinite_metrics_fail_validation():
 m={'pair_count':1000,'pairwise_accuracy':1.0,'loss':float('nan'),'average_margin':1,'counts_by_event':{'PARCEL_RELEASE':1}}
 assert determine_validation_status('assignment',m,m,m,{}, {'validation':{}})=='failed_nonfinite_metrics'
def test_formal_gates_insufficient_and_accuracy():
 cfg={'validation':{'minimum_training_pairs':10,'minimum_validation_pairs':10,'minimum_test_pairs':10,'minimum_validation_pairwise_accuracy':.7,'minimum_test_pairwise_accuracy':.7}}
 tr={'pair_count':1,'pairwise_accuracy':1,'loss':0,'average_margin':1,'counts_by_event':{}}; assert determine_validation_status('assignment',tr,tr,tr,{},cfg)=='failed_insufficient_data'
 good={'pair_count':11,'pairwise_accuracy':.6,'loss':0,'average_margin':1,'counts_by_event':{'PARCEL_RELEASE':11}}; assert determine_validation_status('assignment',good,good,good,{},cfg)=='failed_validation_accuracy'
