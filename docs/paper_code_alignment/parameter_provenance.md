# Parameter Provenance

Every formal medium-instance parameter is listed in `configs/paper/parameter_provenance.yaml` with one category from `literature_adapted`, `project_extension`, `real_input`, `fallback_only`, or `derived`.

Project-extension parameters are not real data. In Phase 0, truck count, truck weight capacity, truck volume capacity, parcel volume distribution, minimum layover, non-service relocation time, truck costs, truck loading time, and truck unloading time are marked `project_extension` unless later provenance supplies a concrete source.

The actual current implementation status is schema-only: provenance validation is implemented, while runtime simulation use of these parameters is deferred.
