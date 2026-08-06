"""Isolated MAPPO policy post-training implementation.

This package deliberately does not alter or wrap the legacy training entry point.
"""

from .config import PostTrainingConfigError, resolve_post_training_config
from .initialization import PolicyInitializationRecord, initialize_policy

__all__ = [
    "PolicyInitializationRecord",
    "PostTrainingConfigError",
    "initialize_policy",
    "resolve_post_training_config",
]
