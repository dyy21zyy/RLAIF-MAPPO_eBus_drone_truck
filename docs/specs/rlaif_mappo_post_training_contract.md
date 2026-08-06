# Isolated RLAIF–MAPPO post-training contract

## Base pretraining

\[
\pi_{\mathrm{base}} = \operatorname{MAPPO}(r^{\mathrm{env}})
\]

The base policy starts from random actor and critic parameters. It is a parent artifact, not a result copied from the original formal experiment.

## Bradley–Terry Reward Model

Assignment preference records have the form \((s,a^+,a^-)\). They train an explicit reward model with

\[
\mathcal L_{\mathrm{RM}}=-\log\sigma\left[r_\phi(s,a^+)-r_\phi(s,a^-)\right].
\]

The validated RM is assignment-only. It is frozen (`requires_grad=False`, evaluation mode) throughout policy optimization.

## RLAIF post-training

\[
\pi_{\mathrm{post}}^{(0)}=\pi_{\mathrm{base}}
\]

\[
r_t^{\mathrm{post}}=r_t^{\mathrm{env}}+\lambda r_\phi(s_t,a_t).
\]

Both child arms strictly load the same parent's actor and centralized-critic weights. Optimizers are constructed **after** loading and start with empty state; this is policy post-training, not checkpoint resume. MAPPO updates the four actors and centralized critic. Only assignment decisions receive the frozen learned-reward contribution.

This is preference-aligned policy post-training. It is **not DPO**, is not SFT, and contains no direct preference policy loss. Preference pairs train the explicit Bradley–Terry RM; asynchronous MAPPO remains the policy optimizer.

## Isolation and fail-closed provenance

Parent checkpoint and manifest hashes, seed, schemas, network dimensions, scenario-bank hash, and reward-scale hash are validated before loading. Child checkpoints record the complete parent lineage. The continuation arm cannot load an RM; the aligned arm accepts only a validated assignment RM and verifies its bytes. Any mismatch aborts before training. Outputs live under `results/formal_post_training`; the original `results/formal` tree and Phase 0–9 workflow remain unchanged.
