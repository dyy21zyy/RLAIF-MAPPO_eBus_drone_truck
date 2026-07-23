# Fix 8h Pre-formal Experiment Gate Plan

These are pre-formal validation runs. They are not final paper experiments.

- Stage orchestration: add `evaluation.preformal_gate` with a single ordered manifest from repository verification through formal launch-plan generation.
- Diagnostic versus strict modes: keep `diagnostic/preformal_diagnostic` separate from strict `formal/preformal`; both are `publication_eligible=false`.
- Artifact inventory: record existence, suffix policy, and SHA-256 hashes before stages consume formal-candidate artifacts.
- Scenario-bank lifecycle: orchestrate existing scenario-bank builders/validators and block downstream stages on missing or invalid banks.
- Reward-scale lifecycle: orchestrate existing reward-reference-scale estimation and validation; strict mode rejects placeholders and diagnostic/smoke artifacts.
- Reward-model validation: verify declared reward checkpoints and preserve the contract that reward models score only selected transitions.
- Short policy training: use configurable reduced episodes/seeds and existing MAPPO/assignment PPO entry points.
- Real benchmark execution: call the existing formal benchmark runner against frozen test banks with transition-count validation.
- Mini ablation: call the existing ablation matrix runner for environment MAPPO and assignment-only RLAIF-MAPPO.
- Mini sensitivity: call the existing sensitivity matrix runner with one factor and two values.
- Resume identity: log invoked commands, inputs, output hashes, and per-stage status for deterministic resume decisions.
- Failure propagation: required downstream stages become `blocked_dependency` unless all required upstream dependencies pass.
- Formal launch-plan generation: emit exact final experiment commands without running 3000×3×100 experiments.
- Runtime artifact isolation: write generated reports only under `results/diagnostic/`, `results/preformal/`, or `results/formal/`.
- Binary-free PR gates: keep generated scenario banks, checkpoints, results, reports, and binary model/data artifacts out of Git.
