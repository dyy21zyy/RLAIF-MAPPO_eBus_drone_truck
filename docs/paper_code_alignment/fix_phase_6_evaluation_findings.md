# Fix Phase 6 Evaluation Findings

Starting commit: `79e7fc83502504e7e7bcc46e38c07134b834c2bd`.

## Confirmed defects

| File | Function | Current behavior | Required behavior | Failing test |
|---|---|---|---|---|
| `evaluation/runner.py` | `EvaluationRunner.run_episode` | RLAIF used a single `RewardModelWrapper` and placed learned reward under assignment. | Use `RewardRegistry` with assignment-only or all four agents and preserve per-agent contributions. | `tests/test_evaluation_uses_reward_registry.py` |
| `evaluation/runner.py` | `EvaluationRunner.run_episode` | Reward wrapper was coupled into evaluation reward shaping path and could affect reported action/reward flow. | Formal actions come only from policy/heuristic; reward models score selected transitions only. | `tests/test_reward_model_does_not_choose_actions.py` |
| `experiments/run_benchmark.py` | `resolve_instance` | Builds an instance during benchmark execution and varies reset seeds. | Formal evaluation iterates frozen scenario-bank artifacts and never regenerates scenarios. | `tests/test_formal_scenario_bank_iteration.py` |
| `evaluation/paired_evaluation.py` | `validate_paired_scenarios` | Paired only on scenario ID sets. | Pair only when scenario ID, instance hash, manifest hash, and artifact hashes match. | `tests/test_paired_scenario_artifact_match.py` |
| `configs/paper/benchmark.yaml` | configuration | Formal config was smoke-sized, used fallback, one seed, and generic reward checkpoint. | Formal config references 100 test scenarios, three seeds, no fallback, distinct policy and reward checkpoints. | `tests/test_formal_experiment_readiness_gate.py` |
| `evaluation/result_schema.py` | `normalize_result` | Missing numeric fields silently defaulted to zero. | Formal metrics must record availability and fail on missing/nonfinite values. | `tests/test_formal_metric_validation_no_silent_missing.py` |
| `experiments/run_paper_ablation.py` | `validate_ablations` | Minimal validation did not enforce actual config differences and separate retrained checkpoints. | Retrained ablations require config differences and separate checkpoint hashes. | `tests/test_formal_ablation_checkpoint_separation.py` |
| `experiments/run_paper_sensitivity.py` | `validate_sensitivity` | Modes existed but did not enforce fixed vs retrained checkpoint semantics. | Keep fixed-policy robustness and retrained-policy sensitivity distinct. | `tests/test_fixed_vs_retrained_sensitivity.py` |
