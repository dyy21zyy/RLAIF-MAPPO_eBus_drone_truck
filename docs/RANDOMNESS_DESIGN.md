# Randomness Design

Formal `original_scale_real_transit` experiments keep transit structure fixed
and restrict randomness to demand, robustness noise, and learning seeds. This
preserves the previous Electric Bus-Drone scale while adding a truck feeder
layer.

## Fixed in formal experiments

Do not randomize the real bus route, `route_id`, `direction_id`, stop sequence,
or planned stop-level timetable when real stop_times are available.

Fixed fields include:

- real `route_id`, `direction_id`, `service_id` when selected;
- real `stop_id`, `stop_name`, coordinates, and `stop_sequence`;
- real planned `arrival_time` and `departure_time` from stop_times;
- selected integrated-station set, chosen from real stops using original scale;
- inherited bus, drone, station, locker, power, battery, parcel-scale, and reward parameters;
- explicit truck-extension defaults unless a config overrides them.

## Random in formal experiments

Seed-controlled random variables:

- parcel release time;
- customer location around the real corridor;
- parcel weight and volume;
- parcel deadline class/slack;
- parcel priority;
- heavy parcel ratio and urgent parcel ratio when configured;
- station base load only if a stochastic-load experiment explicitly enables it;
- RL training seed;
- RLAIF pair sampling/evaluator replay seed.

## Robustness-only randomness

The following should be off in the main benchmark and used only in robustness or
sensitivity experiments:

- bus delay noise;
- truck travel-time noise;
- stochastic station base-load disturbance;
- alternative route windows or integrated-station perturbations.

## Fallback and smoke tests

`configs/shanghai_small.yaml` remains a deterministic fallback/smoke setting.
The original-scale smoke commands use tiny committed fixture CSVs under
`tests/fixtures/transit`. Those fixtures are `fallback_test_only` and must never
be described as real crawled transit data or final experimental evidence.
