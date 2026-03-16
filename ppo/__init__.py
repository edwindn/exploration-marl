"""
Simplified PPO implementation for exploration-marl project.
This package is self-contained and does not depend on stable_baselines3 for core functionality.
"""

from ppo.ppo_algorithm import PPO
from ppo.policy import ActorCriticPolicy, ActorCriticCnnPolicy

__all__ = ["PPO", "ActorCriticPolicy", "ActorCriticCnnPolicy"]
