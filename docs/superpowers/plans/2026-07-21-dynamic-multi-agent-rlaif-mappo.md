# Implementation Plan: Phase 0 Dynamic Multi-Agent RLAIF MAPPO Contract

1. Inspect repository instructions, current docs, config layout, and environment modules.
2. Write failing tests for domain entities, paper configs, parameter provenance, and final contract documentation.
3. Add schema dataclasses/enums and event types for the final dynamic contract.
4. Add small, medium, large paper configs plus formal MAPPO environment-reward and RLAIF configs.
5. Add parameter provenance and validation helpers.
6. Document target architecture and actual current implementation status without marking later phases complete.
7. Run focused tests, full tests, compile checks, diff checks, commit, push, and open a draft PR.

Phase 0 deliberately leaves truck batching, physical-bus circulation, passenger dynamics, station battery decisions, and multi-agent RLAIF as **specified** for Phase 1 and later.
