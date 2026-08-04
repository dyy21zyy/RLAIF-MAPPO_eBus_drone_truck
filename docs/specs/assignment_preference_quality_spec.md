# Assignment Preference Quality Gate v2 — Project Specification

Date: 2026-08-04. This document records the complete normative requirements supplied for the formal RLAIF-MAPPO assignment-preference migration.

## Scope and invariant requirements

RLAIF remains assignment-only; truck, bus, station, MAPPO architecture/hyperparameters, environment dynamics, reward-model thresholds, and existing formal roots are unchanged. Assignment modes are TD (truck direct), TBD (truck–bus–drone), and TLD (truck–locker–drone, no bus). The accepted binary target is exactly **1600** and the total evaluator-attempt budget is **3000**. No production generation occurs as part of implementation and no private data, keys, caches, checkpoints, or generated formal datasets are committed.

## Scheme A migration

Audit all 200 v1 records offline. Retain quality-passing records without an evaluator call or fabricated v2 cache entry; quarantine failures; re-evaluate each quarantine on its exact state/pair with v2; then generate new v2 pairs until exactly 1600 accepted A/B labels exist. Ties, malformed output, semantic rejection, and retries do not count. Labels and original outcomes are never silently changed, and quarantines never train a reward model. The final dataset is scenario-grouped 70/15/15 with legacy split retained as lineage and no scenario leakage.

## Versions and cache identity

Assignment uses `assignment_grounded_preference_v2`, `rlaif_preference_json_v2`, `assignment_preference_quality_v1`, and consequence mode `estimated_candidate_attributes`. Non-assignment and legacy parsing retain v1 compatibility. V2 cache is separate (`preferences/evaluator_cache_v2`) and keys cover prompt/schema/gate/consequence versions, evaluator model/temperature, agent/event, scenario/state/pair identities, and observation/candidate/event schema versions. V1 caches remain read-only and cannot satisfy v2 requests.

## Strict response and evidence contract

Responses contain `preferred` (A/B/equal), finite numeric confidence in [0,1], a nonempty duplicate-free list of canonical criteria, a nonempty list of `{metric, better_candidate}` evidence, and nonempty reason. Hidden learning-method names remain forbidden. Only A/B is usable. Preserve evaluator prose as `raw_evaluator_reason`; official `reason` is deterministically generated from validated support, trade-offs, selected mode, and preferred candidate.

Canonical criteria are: delivery feasibility/time, expected lateness, deadline risk, truck distance/time/capacity, bus wait/linehaul/freight capacity, drone time/feasibility, locker congestion, station power margin, and downstream congestion (stored in snake case). Energy, energy efficiency, emissions/carbon, fuel and electricity consumption are forbidden without direct features. Legacy aliases listed in the task (delivery-time, lateness, locker-load, and power-margin families) migrate only when unambiguous; comma-separated legacy strings may migrate; mappings and ambiguous/unsupported terms fail closed.

A central registry compares lower-is-better delivery time, lateness, truck distance/time, applicable bus wait/linehaul, drone time, and locker congestion; higher-is-better delivery feasibility and station power margin plus capacity/feasibility only where deterministic extractors exist. Tolerance is `1e-9`. Evidence direction is recomputed; at least one item must support the selection; validated trade-offs are allowed.

## Mode grounding and semantics

Truck metrics apply to all modes; bus metrics only TBD; locker/drone/station power only TBD/TLD. Structural zero placeholders are excluded. TD uses only truck. TBD uses truck, bus, station handoff/locker, drone, and may use station battery/power. TLD uses truck, station locker, drone, and may use station battery/power, but no bus. Deterministic lexical/structural checks reject false resource use, feasible-as-infeasible prose, unsupported energy/emission claims, station margin treated as consumption, fabricated simulated-consequence claims, reversed numerical comparisons, and unavailable/inapplicable evidence. No second LLM is used.

A selection is strictly dominated when, over the intersection of available comparable mode-applicable deterministic metrics, it is no better everywhere and worse somewhere. Dominated v2 is rejected and dominated legacy is quarantined; trade-offs remain valid.

## Pair pool, sampling, and audit

Unordered pair types are TD-TBD, TD-TLD, TBD-TBD, TBD-TLD, and TLD-TLD; unknown modes and artificial TD-TD fail closed. Build the complete feasible pool, deterministically select using existing seeded/hash order, reserve total floor 30 and per-split floor 5 only for present types, select all when capacity is below a floor, allocate remainder proportionally, never duplicate, and do not fail for absent TBD-TBD. Accepted legacy counts; quarantine counts only after accepted re-evaluation. Coverage fails only where capacity existed.

Manifests report pool/selected/accepted/quarantined pair counts, pair/split counts, preferred modes, and pool-vs-accepted proportions. Final hard audit requires exactly 1600 binary accepted records, schema/identity/mask/feasibility/split/criteria/evidence/direction/semantics/dominance/coverage/scenario-hash/provenance correctness. It reports labels, modes, pair proportions, confidence, criteria, migration/cache/API/tie/retry/failure counts. Any hard failure blocks reward training.

## Offline CLI and artifacts

`python -m experiments.audit_assignment_preferences` requires input, accepted/quarantine JSONL, JSON report, and Markdown report; optional expected count/SHA/config. It makes zero API calls, never changes input, checks JSON/scope/unique IDs and pairs/mask/feasibility/modes/criteria/semantics/numerical direction/evidence/dominance/label preservation, and fails nonzero for malformed input, count/hash/identity failures, or write failure. Accepted lineage includes gate/status/legacy IDs and versions/split/raw reason/evidence/actions/timestamp. Quarantine includes reasons, IDs/state/pair and both payloads.

Under the new run root create legacy audit outputs, final assignment JSONL and manifest, final quality JSON/Markdown, failed attempts, and v2 cache. Phase 2 order is legacy audit and SHA validation; quarantine re-evaluation; additional v2 generation; final quality audit; reward training; metric/provenance validation; marker/manifest. Marker lineage includes all migration counts and input/audit/checkpoint hashes plus versions; stale v1-only markers are invalid. Preserve clean-git, commit-root, hashing, resume, CUDA, and checkpoint safeguards.

## Configuration, documentation, and verification

Configuration enables Scheme A through `FORMAL_LEGACY_ASSIGNMENT_PREFERENCES` and `FORMAL_LEGACY_ASSIGNMENT_PREFERENCES_SHA256`, failing closed for missing path/file/hash, count other than 200, or mismatch. Tests may disable migration or use synthetic data. Pair types are proportional rather than equal because the feasible pool determines availability. Empty consequence objects are only estimated candidate attributes. New calls can exceed 1400 due to quarantines, ties, invalid output, and retry attempts.

Required verification comprises compileall; focused quality, generation/pipeline, Phase 2/resume/count tests; CLI help; synthetic offline audit showing counts/distributions/zero calls; hard-locker Phase 2 dry plan without `--execute`; diff/status checks; and a broader relevant suite with pre-existing failures explicitly separated. The real audit and 1600-generation are deferred to a new commit-specific root.
