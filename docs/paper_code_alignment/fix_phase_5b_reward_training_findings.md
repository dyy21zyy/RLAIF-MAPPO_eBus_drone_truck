# Fix Phase 5B reward training findings

Confirmed defect: `experiments/train_multi_agent_reward_models.py` wrote metadata-only `smoke_placeholder` JSON instead of neural reward checkpoints. This phase replaces that path with real PyTorch Bradley-Terry optimization and rejects legacy placeholder artifacts.
