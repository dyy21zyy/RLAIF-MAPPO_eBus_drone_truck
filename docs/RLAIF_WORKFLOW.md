# Stage 4 RLAIF Data Workflow

## Scope and safety boundary

Stage 4 collects assignment decision states, creates pairwise prompts, and accepts
preference labels from exactly two provenance-bearing sources:

1. a valid response returned by a configured external API evaluator; or
2. a human- or AI-labeled JSONL file explicitly supplied by the user to replay
   mode.

Stage 4 never turns feasibility, estimated delivery time, lateness, distance,
cost, drone time, locker load, station power margin, or any other objective
feature into `chosen`/`rejected`. Those values are permitted as prompt context
and for choosing which action pair to present, but they are not labels. Stage 4
does not train a reward model or implement PPO/MAPPO.

## Workflow

1. `collect_assignment_states` runs the Stage 3 environment and writes
   `assignment_states.jsonl` with the shared feature schema.
2. `build_ai_preference_prompts` validates the states and writes
   `ai_preference_prompts.jsonl` with bounded cross-mode comparisons.
3. `build_ai_preferences --mode offline` removes any stale preference output and
   writes blank `manual_labels_template.jsonl` and
   `manual_labels_template_small.jsonl`. It creates no preference labels.
4. An approved evaluator fills a template, or API credentials are configured.
5. API or replay mode validates labels and is the only path allowed to create
   `ai_preferences.jsonl`. Invalid API/replay records are written to
   `failed_preferences.jsonl`.

## Label validation

A preference is valid only when `chosen` is `action_a` or `action_b`, `rejected`
is the other action, and confidence is in `[0, 1]`. Valid confidence at or above
`0.6` sets `usable_for_training=true`; lower confidence remains valid but sets
`usable_for_training=false`. Replay never fills missing fields, chooses an
action, or generates additional records for prompts absent from the supplied
label file.

If API configuration is missing, every API response is invalid, or replay has no
valid user-provided record, the workflow removes stale `ai_preferences.jsonl`
and stops with a clear refusal to create rule-based or fabricated labels.

## API configuration can be supplied later

No private API key is committed. The placeholder configuration is
`configs/rlaif_preferences.yaml`:

```yaml
rlaif:
  api_key_env: RLAIF_API_KEY
  api_base_url: ""
  model_name: ""
  temperature: 0.0
  max_retries: 3
```

At runtime, set:

```bash
export RLAIF_API_KEY='...'
export RLAIF_API_BASE_URL='https://provider.example/v1'
export RLAIF_MODEL_NAME='provider-model-name'
```

`RLAIF_API_BASE_URL` and `RLAIF_MODEL_NAME` override config-file values. The
`rlaif.api_key_env` field selects the environment variable containing the key;
the key itself must not be stored in the config. The evaluator sends an
OpenAI-compatible chat-completions request, uses configured temperature and
retry count, and validates the returned JSON before persistence.

An empty API configuration is acceptable during development. API mode then exits
cleanly with instructions to configure the evaluator or use replay mode. It does
not fall back to heuristics, rules, synthetic records, or blank preferences.

## Commands

```bash
python -m experiments.collect_assignment_states \
  --config configs/shanghai_small.yaml --episodes 50 \
  --output data/preference/assignment_states.jsonl --fallback

python -m experiments.build_ai_preference_prompts \
  --states data/preference/assignment_states.jsonl \
  --output data/preference/ai_preference_prompts.jsonl

python -m experiments.build_ai_preferences \
  --mode offline \
  --prompts data/preference/ai_preference_prompts.jsonl

python -m experiments.build_ai_preferences \
  --mode api \
  --config configs/rlaif_preferences.yaml \
  --prompts data/preference/ai_preference_prompts.jsonl \
  --output data/preference/ai_preferences.jsonl

python -m experiments.build_ai_preferences \
  --mode replay \
  --prompts data/preference/ai_preference_prompts.jsonl \
  --labels path/to/user_provided_labels.jsonl \
  --output data/preference/ai_preferences.jsonl

python -m experiments.smoke_test_rlaif_data \
  --config configs/shanghai_small.yaml --fallback
```

## Files by mode

### Offline preparation

The complete offline workflow creates only:

- `data/preference/assignment_states.jsonl`;
- `data/preference/ai_preference_prompts.jsonl`;
- `data/preference/manual_labels_template.jsonl`;
- `data/preference/manual_labels_template_small.jsonl`.

The templates contain prompt/action identity and blank `chosen`, `rejected`,
`confidence`, and reason fields. Offline mode removes stale
`ai_preferences.jsonl` rather than creating an empty or fake label file.

### API mode

With valid configuration and at least one valid evaluator response, API mode
writes `ai_preferences.jsonl` and `failed_preferences.jsonl`. Without valid
configuration or valid responses, it writes no preference file.

### Replay mode

Replay requires `--labels` pointing to a user-provided JSONL file. It writes only
validated supplied labels to `ai_preferences.jsonl` and invalid supplied labels
to `failed_preferences.jsonl`; it does not generate labels for unmatched prompts.

## Reward-model prerequisite

Reward-model training must use only valid records with
`usable_for_training=true`. If `ai_preferences.jsonl` is missing, empty, or has
no usable records, training must stop and instruct the operator to run Stage 4
in configured API mode or replay mode with valid labels. Rules and objective
features are never an alternative label source.

## Known limitations

The fallback instance is synthetic, candidate estimates are myopic, API mode
supports an OpenAI-compatible chat-completions response shape, and Stage 4 makes
no claim about evaluator quality or agreement. Credentials and model selection
remain an operator responsibility at execution time.
