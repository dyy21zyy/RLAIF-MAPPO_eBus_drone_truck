# Known Limitations Before Formal Experiments

The repository is integration-ready for smoke and invariant gates, but it is not a source of final experimental claims.

Missing or blocked artifacts:

- Real transit files are not committed; fallback fixtures are deterministic smoke/test data only.
- Final AI preference labels are not available.
- Validated four-agent reward-model checkpoints are not available.
- Formal trained MAPPO/RLAIF checkpoints are not available.
- Large-scale benchmark, ablation, and sensitivity outputs have not been produced.
- Hardware/runtime validation for long PyTorch training jobs has not been completed.

These gaps must not be replaced with synthetic claims. Any paper table or conclusion must be generated only after the formal runs complete with provenance, configuration hashes, and checkpoint lineage.

## Fix Phase 6 formal evaluation integrity

Formal evaluation now uses frozen scenario-bank manifests. All methods share identical scenario artifacts, and paired comparisons validate scenario ID, instance hash, scenario-manifest hash, and exogenous artifact hashes before comparison. Environment MAPPO, assignment-only RLAIF-MAPPO, and full RLAIF-MAPPO are separate formal method identities with separate policy checkpoints. Full RLAIF evaluation requires four agent-specific reward checkpoints loaded through `RewardRegistry`; assignment-only RLAIF enables only the assignment reward model. Reward models do not select evaluation actions; they validate lineage and score selected transitions for decomposition only.

Formal metrics are fail-closed: missing instrumentation is missing, not zero. Legitimate zero values require an instrumented source and explicit legitimate-zero provenance. Ablations that require retraining require separate checkpoints and actual configuration differences. Sensitivity experiments distinguish fixed-policy robustness from retrained-policy sensitivity and do not aggregate the two modes together by default.

This infrastructure does not claim that the final 100-scenario, three-seed paper benchmark has been executed; formal readiness remains blocked until final frozen banks, trained policies, and validated formal reward checkpoints exist.
