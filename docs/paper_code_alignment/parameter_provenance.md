# Parameter Provenance

Every formal medium-instance parameter is listed in `configs/paper/parameter_provenance.yaml` with one category from `literature_adapted`, `project_extension`, `real_input`, `fallback_only`, or `derived`.

Project-extension parameters are not real data. In Phase 0, truck count, truck weight capacity, truck volume capacity, parcel volume distribution, minimum layover, non-service relocation time, truck costs, truck loading time, and truck unloading time are marked `project_extension` unless later provenance supplies a concrete source.

The actual current implementation status is schema-only: provenance validation is implemented, while runtime simulation use of these parameters is deferred.

## Phase 1 reward-weight provenance

The formal reward block weights added for MAPPO/RLAIF-MAPPO runtime validation are classified as `project_extension` where the source paper does not directly specify the coefficient. They must not be described as literature-derived.

## Fix Phase 3 passenger and station-load provenance

Passenger baseline stop-arrival rates use the paper-derived truncated normal distribution with mean 0.25 passenger/min, standard deviation 0.10 passenger/min, and baseline bounds [0.05, 0.60]. Effective passenger rates are computed as baseline × demand intensity × temporal multiplier and are not clipped back to the baseline maximum. The temporal passenger multipliers are scenario-design assumptions classified as `project_extension` unless an empirical source is later added. Explicit downstream destinations supersede the older independent alighting-probability range.

Station base-load profiles are generated as seeded, piecewise-constant synthetic profiles using Uniform[80, 180] kW at 15-minute intervals by default. They are classified as `project_extension` and are not represented as observed Shanghai load data.
