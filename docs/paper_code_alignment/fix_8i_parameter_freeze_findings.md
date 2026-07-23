# Fix 8i Parameter Freeze Findings

| parameter | current sources | conflicting values | scientific meaning | required frozen source | recurrence test |
|---|---|---|---|---|---|
| Reward reference hashes | `configs/paper/base_*.yaml`, train configs, reward scale config | `REPLACE_WITH_REAL_*` placeholders and diagnostic output paths | Links reward normalization to train scenario bank | `configs/paper/final_experiment_freeze.template.yaml` with `REPLACE_WITH_FINAL_*` blockers | placeholder gate and reward-scale lineage tests |
| MAPPO optimization | `configs/paper/train_mappo_env.yaml` and method overlays | RLAIF overlay files previously omitted inherited values | Ensures reward treatment is the only MAPPO comparison difference | canonical `training_protocol.mappo_optimization` block | MAPPO protocol and method-difference tests |
| Network architecture | method training configs and trainer defaults | architecture can drift if copied per method | Prevents RLAIF methods using larger networks | canonical `network_architecture` block | network-size difference failure test |
| Scenario counts and hashes | benchmark, ablation, sensitivity, preformal configs | preformal diagnostic counts coexist with formal count 100 | Defines formal train/validation/test split sizes and lineage | canonical `scenario_protocol` block | scenario protocol tests |
| Seed semantics | benchmark/training configs | ambiguous seed use for sampling, initialization, shuffling | Reproducible and independent random streams | canonical `seed_protocol.namespaces` block | distinct namespace test |
| RLAIF scope | assignment/all method configs | assignment-only and full extension can be conflated | Defines primary baseline and optional extension | canonical `rlaif_parameters.methods` block | RLAIF scope tests |
| Station actions | environment behavior and paper contract | legacy `charging_start_is_learned` appears in base configs | Station baseline is dispatch/idle; charging is automatic | canonical `scientific_contract.station_baseline_actions` | station baseline test |
| Reward weights | base configs and train config reward blocks | duplicated values and possible silent missing zero | Scientific scalarization of physical reward components | canonical `reward_weights.components` | reward component completeness tests |
| Benchmark row count | launch plan and benchmark config | risk of hardcoded total | Determines expected final evaluation table cardinality | derived from `evaluation_protocol.method_matrix` | row-count derivation test |
| Generated artifacts | results directories | runtime JSON should not enter Git | Immutable freeze/provenance artifacts are generated, not source | `.gitignore` targeted freeze paths | artifact immutability tests |
