# Paper-Code Alignment Decision Log

## 2026-07-20 Four-Agent Solution Method Scope

The user confirmed that this implementation pass must follow the complete
four-agent document alignment. This supersedes the previous Stage 7 two-agent boundary where MAPPO controlled assignment and bus charging only.

The current pass treats the supplied manuscript as source of truth for code
alignment. It must not fabricate preference labels, learned rewards,
checkpoints, real data, benchmark results, ablation results, or sensitivity
results.

## 2026-07-20 Runtime Evidence Boundary

Dependency-light tests and smoke tests can prove code interfaces, masks,
invariants, and checkpoint-loading behavior. They do not prove final paper
performance. Final RLAIF-enabled experiments remain incomplete until real
preference labels, validated reward checkpoints, trained policy checkpoints, and
benchmark/ablation/sensitivity runs exist.
