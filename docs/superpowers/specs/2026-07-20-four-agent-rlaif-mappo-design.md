# Four-Agent RLAIF-MAPPO Alignment Design

## Goal

Align the repository implementation with the confirmed Solution Method manuscript
for a four-agent event-driven RLAIF-MAPPO framework. The target implementation
has event-specific actors for assignment, truck dispatch, bus operation, and
station operation, a shared centralized critic, mask-aware candidate-action
policies, and strict RLAIF provenance boundaries.

## Source Of Truth

The manuscript supplied on 2026-07-20 is the behavior source of truth for this
implementation pass. It supersedes the previous Stage 7 implementation boundary
where MAPPO controlled only parcel assignment and bus charging. Earlier docs
that describe truck and station behavior as deterministic are retained only as
historical stage descriptions until they are updated by this pass.

The implementation must not fabricate real transit data, preference labels,
learned rewards, checkpoints, benchmark results, ablation results, or sensitivity
results. Code smoke tests can validate interfaces and invariants, but they are
not final experiment evidence.

## Current Gap

The current repository already has a staged data pipeline, event-driven
environment, assignment PPO, assignment-only RLAIF workflow, and Stage 7
asynchronous MAPPO code gate. The Stage 7 MAPPO path currently includes:

- assignment actor at parcel-release events;
- bus actor at integrated-station bus-arrival charging events;
- shared centralized critic over `env.get_global_state()`;
- asynchronous transition buffer without inactive-agent rows;
- strict checkpoint loading for learned RLAIF reward.

The confirmed manuscript requires more than this. It requires a heterogeneous
decision process over four agent types:

- assignment agent for delivery-mode and station selection;
- truck agents for direct delivery, bus-terminal feeder service, station-feeder
  service, or idle decisions;
- bus agents for parcel loading at departure and charging at station arrival;
- station agents for drone dispatch and drone-battery charging decisions.

## Requirement Mapping

| ID | Requirement | Primary code area | Validation evidence |
| --- | --- | --- | --- |
| REQ-MDP-FOUR-AGENT | Expose real decision events for assignment, truck, bus, and station agents. | `envs/delivery_env.py`, `envs/state_builder.py` | Environment tests observe all four agent ids without dummy inactive rows. |
| REQ-MDP-CANDIDATES | Every active decision provides candidate actions, candidate features, and a hard feasibility mask. | `envs/state_builder.py`, new candidate helpers | Schema tests validate finite features, stable names, and at least one feasible action per decision. |
| REQ-MDP-TRANSITIONS | Environment actions apply operational state changes for truck dispatch, bus loading/charging, and station drone/battery operation. | `envs/delivery_env.py` | Focused tests verify state deltas, resource accounting, and invariants. |
| REQ-MAPPO-ACTORS | MAPPO owns event-specific actors for assignment, truck, bus, and station. | `training/mappo_networks.py`, `training/mappo_trainer.py` | Network and trainer tests verify actor registry and per-agent optimization metrics. |
| REQ-MAPPO-CANDIDATE-POLICY | Actors score candidate feature rows with masked softmax rather than relying only on fixed action heads. | `training/mappo_networks.py` | Mask tests verify infeasible candidates receive zero probability and are never sampled. |
| REQ-MAPPO-BUFFER | The rollout buffer stores only activated agents and supports event-time discounting. | `training/mappo_buffer.py` | Buffer tests verify four agent ids, no inactive rows, episode resets, and time-delta discount factors. |
| REQ-RLAIF-SCOPE | Learned preference reward is optional, checkpoint-backed, and never rule-generated. | `training/reward_model_wrapper.py`, `training/mappo_async.py`, `rlaif/` | Tests verify disabled mode loads no checkpoint and enabled mode fails without a valid checkpoint. |
| REQ-DOC-TRACE | Alignment docs record satisfied, deferred, and blocked paper requirements. | `docs/paper_code_alignment/` | Traceability check and documentation tests cover guardrail wording. |

## Architecture

### Environment Decision Surface

`DynamicDeliveryEnv` remains the single event-driven simulator. It should expose
one pending decision at a time through the existing `reset()` and `step(action)`
API to preserve compatibility with current smoke scripts. The observation
payload is extended rather than replaced:

- `agent_id`: one of `assignment`, `truck`, `bus`, `station`, or `terminal`;
- `event_type`: one of `PARCEL_RELEASE`, `TRUCK_AVAILABLE`, `BUS_DEPARTURE`,
  `BUS_ARRIVAL`, `STATION_OPERATION`, or `TERMINAL`;
- `entity_id`: parcel, truck, trip, station, or compound operational id;
- `features`: local observation vector for the active agent;
- `candidate_actions`: structured action descriptors for audit and RLAIF prompt
  construction;
- `candidate_features`: numeric feature rows aligned with `candidate_actions`;
- `candidate_feature_names`: stable ordered feature names for the active agent;
- `action_mask`: hard feasibility mask aligned with candidates.

The environment still owns physical transitions, resource ledgers, reward
components, and invariants. The policy only selects an action index from the
provided candidate set.

### Agent Responsibilities

The assignment agent decides high-level parcel mode and target station:

- `TO`: truck direct delivery;
- `BLD_h`: truck feeder to a bus-terminal-compatible route followed by bus and
  drone delivery through station `h`;
- `TLD_h`: truck feeder to station `h` followed by station drone delivery.

Truck agents decide which ready truck task to execute:

- direct customer delivery task;
- feeder task to a freight-enabled bus terminal or transfer stop;
- station feeder task;
- idle when no feasible task is useful or available.

Bus agents cover two event roles under one shared bus policy:

- departure loading: select a bounded candidate subset of waiting parcels for a
  freight-enabled trip while respecting capacity and release constraints;
- station-arrival charging: select a configured charging duration while
  respecting charger availability as a hard mask and station overload as a soft
  cost.

Station agents decide station operation:

- dispatch a feasible waiting parcel with an available drone and battery;
- start or defer drone-battery charging when the station has depleted batteries
  and power resources;
- idle when no feasible station action exists.

### Candidate Features And Masks

Candidate generation is deterministic and state-derived. It may use objective
features such as travel time, capacity, deadline slack, expected lateness,
locker load, drone availability, battery availability, and station power margin.
These features are context and policy inputs only. They must never be converted
into preference labels.

Masks enforce hard physical feasibility: parcel release status, truck capacity
and availability, bus route and freight capacity, charger availability, drone
payload and range, drone and battery availability, and resource conflicts.
Soft operational penalties such as power overload, locker overflow, delay, and
cost remain in the reward ledger.

### MAPPO Training

The Stage 9 trainer generalizes the current Stage 7 trainer:

- actor registry keyed by agent type;
- candidate-scoring actor class that scores each candidate row using local
  observation, candidate features, and an event-type embedding;
- shared centralized critic over aggregate global state;
- asynchronous rollout buffer with one transition per activated agent decision;
- event-time discount computed from adjacent event times using
  `gamma ** (delta_time / reference_time_unit)`;
- PPO actor updates grouped by agent type;
- critic update over the complete event stream.

Same-type agents share actor parameters. Execution remains decentralized: only
the active agent observation, candidate features, and mask are needed online.

### RLAIF Boundary

The default code gate keeps `rlaif.enabled: false`, requiring no reward-model
checkpoint. When enabled, the trainer may add learned preference reward only
through a valid Stage 5 checkpoint loaded by `RewardModelWrapper`. A missing,
invalid, or dimension-incompatible checkpoint is an error. No rule score,
heuristic preference, objective-feature label, evaluator reason text, or blank
template may become a learned reward.

Initial Stage 9 code may keep the existing assignment reward model for learned
reward and mark truck/station preference reward as blocked until multi-agent
preference datasets and checkpoints are available. This is acceptable only when
recorded in `docs/paper_code_alignment/decision_log.md` and
`validation_report.md`.

## Data Flow

1. The environment advances automatic events until an operational decision event
   is pending.
2. The active agent observation exposes local features, candidate actions,
   candidate features, and a feasibility mask.
3. The MAPPO actor registry selects the actor for the active agent type.
4. The actor scores masked candidates and samples or selects one action.
5. The environment applies the selected action, advances automatic events, and
   returns environment reward and audit information.
6. The rollout buffer records the activated agent transition only.
7. The trainer computes event-time returns and advantages using the centralized
   critic.
8. PPO updates actor parameters per agent type and critic parameters globally.

## Testing Strategy

All behavior changes use TDD. Focused tests come before production code:

- environment tests for four-agent decision exposure and state transitions;
- schema tests for candidate actions and candidate feature names;
- network tests for candidate-scoring masked actors;
- buffer tests for four-agent storage and event-time discounting;
- trainer smoke tests for actor registry, finite losses, checkpoint round trip,
  and no fabricated RLAIF reward;
- documentation tests for traceability and no-fabrication guardrails.

The final verification pass runs focused tests first, then dependency-light
project gates. PyTorch runtime gates are reported as skipped or deferred if the
installed runtime is missing or broken.

## Non-Goals

This pass does not produce final paper benchmark results, train a final reward
model, create preference labels from rules, commit runtime checkpoints, commit
private raw transit data, or claim real-world operational performance from smoke
tests.

## Open Decisions Recorded As Scope

The user confirmed on 2026-07-20 that the implementation should follow the
complete four-agent document alignment rather than the earlier two-agent Stage 7
boundary. Any remaining preference-reward coverage beyond assignment requires
real API or replay labels and a validated checkpoint before it can be marked
runtime-complete.
