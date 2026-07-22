# Step 5 Formal Benchmark Findings

| file | function | current behavior | required behavior | failing test |
|---|---|---|---|---|
| `experiments/run_paper_benchmark.py` | `run_benchmark` | Loaded frozen instances but did not construct `DynamicDeliveryEnv`, call `reset()`, or call `step()`. | Execute a real rollout per successful row. | `tests/test_real_benchmark_executes_environment.py` |
| `experiments/run_paper_benchmark.py` | `run_benchmark` | Wrote fixed zero environment/RLAIF metrics and marked rows successful. | Populate metrics from runtime instrumentation and reject placeholder success rows. | `tests/test_real_benchmark_formal_metrics.py` |
| `evaluation/formal_policy_registry.py` | `validate_policy_checkpoint` | Lacked complete diagnostic/formal classification and method-scope checks. | Fail closed for wrong algorithm, scope, seed, and diagnostic/smoke checkpoints in formal mode. | `tests/test_real_benchmark_checkpoint_validation.py` |
| `evaluation/formal_metric_validation.py` | `validate_formal_metrics` | Used one generic error and did not expose specific missing/non-finite/reconciliation failures. | Raise explicit formal metric exceptions and preserve legitimate-zero provenance. | `tests/test_real_benchmark_missing_metric_failure.py` |
| `evaluation/paired_evaluation.py` | pairing validation | Pairing could pass with incomplete artifact identity. | Pair only identical scenario, instance, manifest, bank, and artifact hashes. | `tests/test_real_benchmark_scenario_pairing.py` |
