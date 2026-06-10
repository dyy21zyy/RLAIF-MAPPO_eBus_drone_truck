# Stage 4 RLAIF Data Workflow

## Scope and boundaries

Stage 4 collects assignment decision states and creates versioned pairwise prompts
for an external AI evaluator. It does **not** create rule labels, train a reward
model, calculate a rule reward, or implement PPO/MAPPO. Offline mode is the safe
default and never invents labels.

## Workflow

1. `collect_assignment_states` builds a Stage 2 instance, runs the Stage 3
event-driven environment with random feasible assignment actions, uses action
zero as the deterministic bus baseline, and records every assignment event.
2. `build_ai_preference_prompts` validates those states and selects at most three
cross-mode comparisons per state: TD/nearest feasible TBD, TD/nearest feasible
TLD, and best-estimated TBD/best-estimated TLD.
3. `build_ai_preferences` either leaves prompts unlabeled (`offline`), sends them
to a configured external endpoint (`api`), or validates supplied labels
(`replay`). Invalid records are retained in `failed_preferences.jsonl`.

## Schemas

### Assignment state (`feature_schema_version: v1`)

Each JSONL record contains `state_id`, episode/time fields, the shared normalized
assignment vector from `envs/state_builder.py`, parcel attributes, aggregate
system state, per-integrated-station state, the complete `1 + 2H` action list,
action mask, and objective candidate estimates. Candidate fields are action ID
and name, feasibility, estimated delivery/lateness, truck distance/time, bus
wait/line-haul time, drone time, projected locker load, station power margin,
and infeasibility reasons. Preference scores are forbidden.

### Prompt (`prompt_version: v1`)

Each record contains `prompt_id`, `state_id`, `action_a`, `action_b`, the prompt
version, JSON-only prompt text, and selection metadata. The evaluator must return
`chosen`, `rejected`, `confidence`, and `reason`; chosen/rejected must be the two
compared actions.

### AI preference

Validated labels include pair IDs, chosen/rejected, confidence, reason, prompt
version, evaluator model, temperature, UTC creation time, parser status,
validation status, training usability, and the raw response. Confidence below
`0.6` is valid and retained but has `usable_for_training=false`.

## Modes

- **Offline:** persists prompts in the prompt-building step and creates no
  preference file or labels.
- **API:** requires `RLAIF_API_URL` and `RLAIF_API_KEY`, uses an OpenAI-compatible
  chat response shape, defaults to temperature `0.0`, retries at most three
  times, and records exhausted/invalid responses as failures.
- **Replay:** matches manual labels by `prompt_id` (or exact state/action pair),
  validates them, writes clean preferences, and records every invalid/unmatched
  label as a failure.

## Commands and Stage 4 gate result

```bash
python -m experiments.collect_assignment_states --config configs/shanghai_small.yaml --episodes 50 --output data/preference/assignment_states.jsonl --fallback
python -m experiments.build_ai_preference_prompts --states data/preference/assignment_states.jsonl --output data/preference/ai_preference_prompts.jsonl
python -m experiments.build_ai_preferences --mode offline --prompts data/preference/ai_preference_prompts.jsonl
python -m experiments.build_ai_preferences --mode replay --prompts data/preference/ai_preference_prompts.jsonl --labels path/to/manual_labels.jsonl --output data/preference/ai_preferences.jsonl
python -m experiments.smoke_test_rlaif_data --config configs/shanghai_small.yaml --fallback
```

The 50-episode fallback run produced 3,000 assignment states and 3,150 prompts.
Offline mode produced zero labels and used no external API. The smoke gate
produced 60 states and 63 prompts.

## Manual labeling without an API

Open each `ai_preference_prompts.jsonl` record, give `prompt_text` to an approved
evaluator, and save one JSONL label with its `prompt_id`, `chosen`, `rejected`,
`confidence`, and `reason`. Then run replay mode. Do not infer missing responses
or convert objective estimates into labels.

## Known limitations

The fallback instance is synthetic; episodes currently revisit the same instance
with different random feasible assignment choices; objective estimates are
myopic and do not simulate downstream competition; API mode supports one generic
OpenAI-compatible HTTP response shape and requires operator-supplied credentials;
and no preference quality, agreement, or reward-model claim is made in Stage 4.
