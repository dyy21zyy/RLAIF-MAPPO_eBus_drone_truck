# RLAIF Workflow Template

> Status: future work. RLAIF is explicitly not implemented in Stage 1.

## Planned boundaries

1. Freeze and validate the environment reward and baseline policies.
2. Define trajectory summaries that omit sensitive/private data.
3. Generate auditable preference labels under a versioned rubric.
4. Split preference data by scenario/seed to avoid leakage.
5. Train and evaluate a reward model against held-out comparisons.
6. Combine environment and learned reward using an explicit coefficient.
7. Report reward hacking, distribution shift, and safety/invariant checks.

## Required provenance

Record the prompt/rubric version, model/provider version, generation parameters,
raw comparison identifiers, filtering decisions, annotator agreement, dataset
hash, and reward-model checkpoint.

## Evaluation template

- Pairwise preference accuracy
- Calibration and uncertainty
- Agreement with environment constraints
- Robustness across seeds and scenario sizes
- Ablation with `rlaif_lambda = 0`
- Qualitative audit of high-scoring trajectories
