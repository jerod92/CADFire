"""Training sub-package: PPOTrainer, RolloutBuffer, CheckpointManager."""

# Lazy imports to avoid circular dependency between ppo -> checkpoint -> __init__ -> ppo
def __getattr__(name):
    if name == "PPOTrainer":
        from cadfire.training.ppo import PPOTrainer
        return PPOTrainer
    if name == "RolloutBuffer":
        from cadfire.training.rollout import RolloutBuffer
        return RolloutBuffer
    if name == "CheckpointManager":
        from cadfire.training.checkpoint import CheckpointManager
        return CheckpointManager
    raise AttributeError(f"module 'cadfire.training' has no attribute {name}")
