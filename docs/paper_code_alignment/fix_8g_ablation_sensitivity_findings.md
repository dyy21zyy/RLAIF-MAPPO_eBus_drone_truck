# Step 6B ablation/sensitivity inspection findings

## Findings

| File | Function | Current behavior | Required behavior | Failing test |
|---|---|---|---|---|
| `experiments/run_paper_ablation.py` | `validate_ablations` | Loads YAML and validates that entries have name, config switch, and checkpoint; never trains or evaluates. | Matrix runner must resolve training configs, invoke MAPPO training, invoke benchmark evaluation, write statuses, failures, paired results, and aggregation. | `tests/test_ablation_runner_executes_training.py` |
| `experiments/run_paper_sensitivity.py` | `validate_sensitivity` | Loads YAML and validates mode/dimension labels only. | Sensitivity runner must execute fixed-policy robustness and retrained-policy sensitivity as separate protocols. | `tests/test_sensitivity_fixed_policy_protocol.py` |
| `evaluation/statistics.py` | `summarize_metric` | Summarizes rows generically and can report one-sample intervals as ordinary summaries. | Experiment aggregation must separate compatibility groups and mark insufficient paired samples. | `tests/test_experiment_aggregation_compatibility.py` |
| `evaluation/paired_evaluation.py` | `validate_paired_scenarios` | Validates benchmark scenario pairing but has no sensitivity scenario-family identity model. | Scenario-family master seeds and artifact identity must be preserved for sensitivity pairing. | `tests/test_sensitivity_scenario_family_pairing.py` |
| `experiments/train_mappo_async.py` | `main` | Provides real MAPPO training entry point over resolved configs and scenario banks. | Orchestration should reuse this entry point, not reimplement MAPPO. | `tests/test_ablation_runner_executes_training.py` |
| `experiments/run_paper_benchmark.py` | `run_benchmark` | Provides real frozen-bank benchmark execution and paired benchmark outputs. | Orchestration should create resolved benchmark configs and invoke this runner. | `tests/test_ablation_runner_executes_benchmark.py` |
| `experiments/build_scenario_bank.py` | `build_bank` | Builds frozen scenario bank manifests with content hashes. | Diagnostic preparation should reuse this builder and keep generated banks ignored. | `tests/test_experiment_artifacts_are_temporary.py` |
| `training/config_resolver.py` | `resolve_mappo_training_config` | Supports seed and output-root overrides and scenario-bank fields. | Matrix runners should inject bank lineage and per-job output paths. | `tests/test_experiment_job_resume_identity.py` |

## Confirmation

The prior paper ablation/sensitivity wrappers did not call `experiments.train_mappo_async` or `experiments.run_paper_benchmark`; they were configuration validators. Step 6B replaces that placeholder layer with real matrix orchestration.
