# Paper-Code Alignment Decision Log

## 2026-07-11: Source of Truth

Default decision: use the confirmed paper manuscript as the source of truth when
the manuscript and code conflict.

Current limitation: no final `.tex` source was provided in this checkout, so the
first pass records unresolved manuscript-dependent claims as
`blocked-by-user-decision`.

## 2026-07-11: No Fabrication Rule

Do not fabricate real data, labels, rewards, checkpoints, or experiment results.

Consequences:

- Synthetic or fixture transit data may support smoke tests only.
- Objective features may appear in prompts but must not become RLAIF labels.
- Rule-based baselines must not become preference supervision.
- Missing PyTorch or missing checkpoints must skip or fail clearly.
- Smoke outputs are code-validation artifacts, not final experimental evidence.

## 2026-07-11: Runtime Boundary

The local Windows runtime may have a broken or unavailable PyTorch package. The
implementation treats that as PyTorch runtime unavailable. Stage 5, Stage 6,
Stage 7, Stage 9, and final Stage 8 learned-policy experiments remain deferred
until a working PyTorch environment and valid checkpoints are supplied.

## Pending User Decisions

| Decision | Default until confirmed |
| --- | --- |
| Final manuscript source | Use existing repo docs and prior paper draft scope as a provisional contract. |
| TBD truck feeder semantics | Preserve current terminal-to-bus semantics unless superseded by a later confirmed formulation. |
| RLAIF scope | Preserve assignment-only learned reward until real multi-agent preference labels and checkpoints exist. |
| Bus charging action unit | Preserve current seconds action set `[0, 15, 30, 45, 60, 75, 90, 105, 120]`. |
| Real transit CSVs | Treat unavailable local files as operator-provided private data, not committed data. |

## 2026-07-20: Four-Agent Solution Method Scope

The user confirmed that this implementation pass must follow the complete
four-agent document alignment. This supersedes the previous Stage 7 two-agent boundary
where MAPPO controlled assignment and bus charging only.

The current pass treats the supplied manuscript as source of truth for code
alignment. It must not fabricate preference labels, learned rewards,
checkpoints, real data, benchmark results, ablation results, or sensitivity
results.

## 2026-07-20: Runtime Evidence Boundary

Dependency-light tests and smoke tests can prove code interfaces, masks,
invariants, and checkpoint-loading behavior. They do not prove final paper
performance. Final RLAIF-enabled experiments remain incomplete until real
preference labels, validated reward checkpoints, trained policy checkpoints, and
benchmark/ablation/sensitivity runs exist.
