from pathlib import Path
from evaluation.preformal_part3_gates import readiness_status

def test_readiness_status_examples():
    stages={'benchmark_execution':'passed','repository_verification':'passed'}
    assert readiness_status(stages,run_classification='diagnostic')=='PREFORMAL_DIAGNOSTIC_PASSED'
    assert readiness_status({'benchmark_execution':'failed'},run_classification='formal')=='BLOCKED_BENCHMARK'
    assert readiness_status({'benchmark_execution':'passed','environment_mappo_training':'passed','assignment_rlaif_mappo_training':'blocked_missing_artifact'},run_classification='formal')=='PREFORMAL_ENVIRONMENT_PATH_PASSED_RLAIF_BLOCKED'
