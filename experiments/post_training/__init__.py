"""Isolated formal MAPPO post-training experiment support."""

METHODS = (
    "mappo_env_post_base",
    "mappo_env_post_continued",
    "mappo_rlaif_assignment_post",
)
SEEDS = (1, 2, 3)
OUTPUT_ROOT = "results/formal_post_training"
