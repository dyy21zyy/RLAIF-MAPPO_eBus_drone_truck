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
