# Step 6A reward reference scale plan

Starting `main` SHA: `3703a9b84636d9b58769aec135461e2199f184cd`.

- Raw reward-component collection: collect `DynamicDeliveryEnv.raw_cost_components` per real rollout before reference-scale division, reward weighting, sign conversion, or RLAIF addition.
- Training-bank-only enforcement: require frozen scenario-bank manifests with `split: train`, verify bank hash and scenario artifact hashes, and reject validation/test/unknown splits.
- Reference policy suite: add deterministic versioned truck-direct, integrated-rule, and coverage policies using only feasible action-mask entries.
- Robust scale estimation: use positive raw episode totals with percentile/median/mean/maximum methods and formal percentile-95 default.
- Zero/unexercised/missing handling: classify each canonical component as observed-positive, instrumented-zero, unexercised, or missing; require documented overrides only for instrumented-zero components.
- Artifact schema and hashing: write version-1 `reward_reference_scales` artifacts with component order, statistics hashes, resolved config hash, code commit, training-bank lineage, and a canonical hash excluding only `artifact_hash`.
- Formal lineage validation: formal loaders reject diagnostic artifacts, placeholder hashes, wrong training-bank hashes, missing/unexercised components, failed validation, nonfinite, or nonpositive scales.
- Temporary diagnostic artifacts: generate diagnostic banks/configs/artifacts under `results/diagnostic/reward_scales/` or pytest `tmp_path` only.
- Binary-free PR gates: ignore generated reward-scale outputs and run staged generated-artifact and binary-extension gates before commit/PR.
