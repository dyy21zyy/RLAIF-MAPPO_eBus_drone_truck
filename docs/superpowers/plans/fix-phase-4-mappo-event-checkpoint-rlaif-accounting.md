# Fix Phase 4 implementation plan

Starting main SHA: `2ad7e1251bea2273d0dfd9c02d7f208ea503dc5b` (network fetch to GitHub was blocked by HTTP 403 in this environment).

Plan:
1. Add a canonical decision-event schema and reject automatic events at MAPPO boundaries.
2. Thread canonical event IDs through actor action selection, rollout transitions, minibatches, PPO evaluation, and checkpoints.
3. Replace helper-only event encoding with learned actor event embeddings.
4. Preserve one shared bus actor for terminal departure and station-arrival decisions.
5. Version checkpoint metadata and distinguish environment, assignment RLAIF, and all-agent RLAIF algorithms.
6. Add structured per-agent RLAIF contribution accounting with normalize -> clip -> lambda -> add order.
7. Document RED/GREEN focused tests and run smoke/full verification.

Out of scope: Bradley-Terry reward-model training and formal RLAIF experiments (Fix Phase 5+).
