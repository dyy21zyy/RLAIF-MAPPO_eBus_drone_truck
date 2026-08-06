# Hard-locker RLAIF–MAPPO post-training on AutoDL

This isolated workflow writes to `results/formal_post_training`. Run a dry plan first (it invokes no evaluator, training, or benchmark):

```bash
python -m experiments.run_hard_locker_post_training_experiment --through 11 --device cuda:0
```

After merge and artifact review, execute/resume one phase at a time:

```bash
python -m experiments.run_hard_locker_post_training_experiment --execute --through 0 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 1 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 2 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 3 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 4 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 5 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 6 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 7 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 8 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 9 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 10 --device cuda:0
python -m experiments.run_hard_locker_post_training_experiment --execute --resume --through 11 --device cuda:0
```

Phase 3 trains three random-initialized base MAPPO policies. Phase 4 collects base-policy assignment preference candidates. **Phase 5 calls the AI evaluator and trains/validates the Bradley–Terry RM**, so do not execute it without credentials and approval. Phase 7 trains environment-reward continuations; Phase 8 trains assignment-only RLAIF–MAPPO children. Phase 10 runs the 900-row common-scenario benchmark. Phase 11 performs paired statistical analysis. Phases 6 and 9 are fail-closed fairness, lineage, checkpoint, and freeze gates.

The confirmatory primary comparison is `mappo_rlaif_assignment_post` versus `mappo_env_post_continued`: per seed, both load the exact same base checkpoint and use equal budgets/hyperparameters. The optional cross-experiment comparison utility is exploratory and never feeds the primary analysis.
