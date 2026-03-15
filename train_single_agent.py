import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env

from env import NavEnv
from agent import IMPALA


class IMPALAExtractor(BaseFeaturesExtractor):
    """Wraps the IMPALA CNN from agent.py as an SB3 feature extractor."""

    def __init__(self, observation_space):
        features_dim = IMPALA.config["feature_size"]
        super().__init__(observation_space, features_dim)
        self.impala = IMPALA()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 normalizes uint8 images to float32 [0,1] and transposes HWC→CHW
        # so observations arrive as (batch, C, H, W) — IMPALA.forward handles both
        return self.impala(observations)


if __name__ == "__main__":
    env = NavEnv(size=200, action_type="discrete")
    check_env(env, warn=True)

    policy_kwargs = dict(
        features_extractor_class=IMPALAExtractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_nav_agent")
    print("Training complete. Model saved to ppo_nav_agent.zip")
