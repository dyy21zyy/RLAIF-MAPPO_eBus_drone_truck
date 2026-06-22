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

# Stage 5 Learned Assignment Reward Model

Stage 5 trains a scalar `AssignmentRewardModel` for one `(state, action)` pair.
An action-ID embedding is concatenated with normalized Stage 4 state and action
features and processed by a ReLU MLP. For each approved pair, training minimizes
`-logsigmoid(r_chosen - r_rejected).mean()`. Reason text is not used.

Only `validation_status=valid`, `usable_for_training=true` records (by default),
valid opposite choices, and confidence at or above the configured threshold are
accepted. There is no rule-label, synthetic-label, or objective-feature fallback.
If the preference file is missing, empty, or yields no usable records, training
prints: `No usable AI/human preference labels found. Run Stage 4 in API mode or
replay mode with valid labels before training the reward model.`

Run training and evaluation with:

```bash
python -m experiments.train_reward_model --config configs/train_reward_model.yaml --data data/preference/ai_preferences.jsonl
python -m experiments.evaluate_reward_model --config configs/train_reward_model.yaml --checkpoint results/checkpoints/reward_model.pt
python -m experiments.smoke_test_reward_model
```

The checkpoint contains model/config/schema metadata, dimensions, the action
mapping, training-only feature normalization statistics, split metrics, and the
mean and population standard deviation of chosen plus rejected training scores.
A later stage may normalize a score as
`(r_ai - reward_mean) / (reward_std + 1e-6)`; Stage 5 does not call PPO.

The current ten-label replay set supports pipeline validation only. Its
validation/test partitions contain one label each and cannot establish reward
model quality or generalization.

## Stage 5 gates and Stage 6 handoff

Stage 5 has two independent gates:

- **Code Gate (passed in the current dependency-light environment):** verifies the
  source interfaces, dataset filtering and missing-label behavior, graceful
  missing-PyTorch handling, clean test skips, absence of fabricated/rule-based
  labels, ignored runtime artifacts, and all non-runtime checks.
- **Runtime Gate (deferred):** proves actual reward-model training, checkpoint
  loading, and evaluation in an environment with `torch>=2.0,<3.0`. Run exactly:

  ```bash
  python -m experiments.smoke_test_reward_model
  python -m experiments.train_reward_model --config configs/train_reward_model.yaml --data data/preference/ai_preferences.jsonl
  python -m experiments.evaluate_reward_model --config configs/train_reward_model.yaml --checkpoint results/checkpoints/reward_model.pt
  python -m pytest -q
  ```

A missing PyTorch installation does not fail the Code Gate. Runtime commands must
report the required dependency instead of emitting an import traceback, and tests
that exercise the PyTorch model use `pytest.importorskip("torch")`.

Stage 6 code development may proceed before the Runtime Gate. Stage 6 must support
`rlaif_enabled=false` for dependency-light code smoke tests. It may accept
`rlaif_enabled=true` only when a valid, trained `reward_model.pt` is available; final
RLAIF-enabled PPO/MAPPO experiments remain deferred until then.

# Stage 6 Assignment PPO Reward Integration

Stage 6 has one learned actor-critic: the assignment policy at parcel-arrival
choices. It does not learn bus charging and does not introduce MAPPO or a
centralized critic. Bus events are resolved by a configured fixed baseline
(`no_charge`, `uniform_30`, or `battery_threshold`) while Stage 3 continues to
handle truck, drone, locker, battery, and station operations deterministically.

When `rlaif.enabled: false`, the assignment transition uses
`R_total = R_env_assignment`; constructing the trainer does not load or require a
reward-model checkpoint. When enabled, it uses
`R_total = R_env_assignment + lambda_rlaif * R_RLAIF`. `R_RLAIF` comes only from
`RewardModelWrapper.score(state_features, action_features, action_id)`, using the
Stage 5 model weights and saved state-feature, action-feature, and score
normalization. AI reason text is not an input. A missing, malformed, or
incompatible checkpoint is an error—objective rules must never be substituted,
because doing so would change the provenance and meaning of the learned reward.

The Stage 6 implementation initially attributes finite environment reward from an
assignment event through the next assignment event (including deterministic
intervening processing under the fixed bus baseline). Optimized episode-end
ledger attribution remains future work and is not claimed here. Final
RLAIF-enabled PPO experiments remain blocked until the Stage 5 Runtime Gate has
produced and validated `reward_model.pt` in a PyTorch environment.

# Stage 7 Asynchronous MAPPO Integration

Stage 7 has two decentralized, mask-aware actors and one shared centralized
critic. It is asynchronous: a parcel-arrival event produces one assignment
transition, while an integrated-station bus-arrival event produces one bus
transition. There is no inactive-agent padding and no simultaneous joint action.
The assignment action IDs remain `0=TD`, `1..H=TBD_h`, and `H+1..2H=TLD_h`; bus
action IDs map to `[0, 15, 30, 45, 60, 75, 90, 105, 120]` seconds. The critic sees
the fixed-size vector returned by `env.get_global_state()`.

When `rlaif.enabled: false`, the assignment reward is its event-to-event environment
reward and the wrapper does not access a checkpoint. When enabled, the wrapper must
load a valid Stage 5 `reward_model.pt`, and the normalized learned score is added
only to assignment transitions. Bus transitions always use environment reward.
There is deliberately no rule-based, reason-text, or fabricated reward fallback.
The current event-to-event reward attribution and logged cumulative decomposition
are first-implementation limitations, not final delayed credit assignment.

```bash
python -m experiments.train_mappo_async --config configs/train_mappo_async.yaml
python -m experiments.evaluate_mappo_async --config configs/train_mappo_async.yaml --checkpoint results/checkpoints/mappo_async.pt
python -m experiments.smoke_test_mappo_async
```

PyTorch is required to execute actors, critic, PPO updates, and checkpoint loading.
Final RLAIF-enabled MAPPO runs remain blocked until the Stage 5 Runtime Gate has
validated a trained checkpoint in such an environment.

## Stage 8 evaluation boundary

Stage 8 registers `assignment_ppo_rlaif` and `mappo_async_rlaif`, but runs them only when their learned-policy checkpoint, a valid Stage 5 `reward_model.pt`, and PyTorch are available. Disabled-RLAIF baselines do not touch the reward checkpoint. The rule-based Stage 8 policy must never be converted into preference labels or reward supervision: doing so would be a fabricated, circular substitute for AI feedback.

Final RLAIF-enabled benchmark, ablation, and sensitivity runs remain deferred until the Stage 5 Runtime Gate and Stage 6/7 runtime training have completed.

Runtime readiness therefore requires a working PyTorch environment, a Stage 5
Runtime Gate-trained `reward_model.pt`, a trained `assignment_ppo.pt` and/or
`mappo_async.pt`, and the configured benchmark/ablation/sensitivity runs. Code-gate
smoke artifacts do not satisfy any of these runtime prerequisites.
