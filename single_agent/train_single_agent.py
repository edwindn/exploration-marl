import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
from stable_baselines3 import A2C

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import gymnasium as gym
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import NavEnv
from utils.networks import IMPALA
from single_agent.logger import WandbCallback


class IMPALAExtractor(BaseFeaturesExtractor):
    """Wraps the IMPALA CNN from agent.py as an SB3 feature extractor."""

    def __init__(self, observation_space, config_path: str = "train_config.yaml"):
        # Load config to get feature_size
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        backbone_config = train_config['agent']['backbone']
        features_dim = backbone_config['feature_size']

        super().__init__(observation_space, features_dim)

        # Extract input dimensions from observation space (H, W, C)
        inp = observation_space.shape
        input_dims = (inp[1], inp[2], inp[0])
        self.impala = IMPALA(input_dims, backbone_config)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 normalizes uint8 images to float32 [0,1] and transposes HWC→CHW
        # so observations arrive as (batch, C, H, W) — IMPALA.forward handles both
        return self.impala(observations)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train A2C agent on navigation environment")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Load configuration
    root_dir = Path(__file__).parent.parent
    with open(root_dir / "single_agent" / "train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(root_dir / "environment" / "env_config.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    # Create environment
    env = NavEnv()
    check_env(env, warn=True)

    # Configure policy to use IMPALA feature extractor
    policy_kwargs = dict(
        features_extractor_class=IMPALAExtractor,
        net_arch=config["agent"]["policy"]["net_arch"],
        activation_fn=nn.ReLU,
    )

    # Initialize A2C
    a2c_config = config["a2c"]
    model = A2C(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=a2c_config["n_steps"],
        learning_rate=a2c_config["learning_rate"],
        gamma=a2c_config["gamma"],
        gae_lambda=a2c_config["gae_lambda"],
        ent_coef=a2c_config["ent_coef"],
        vf_coef=a2c_config["vf_coef"],
        verbose=a2c_config["verbose"],
    )

    # Configure logger to disable stdout table output
    # Only keep csv and tensorboard if needed, remove 'stdout'
    new_logger = configure(folder=None, format_strings=["csv"])
    model.set_logger(new_logger)

    # Create wandb callback with all config for logging (if enabled)
    callback = None
    if not args.disable_wandb:
        wandb_config = {
            "algorithm": "A2C",
            **env_config,
            **config["a2c"],
            **config["agent"]["policy"],
            **config["training"],
        }

        # Get GIF logging config
        logging_config = config["logging"]
        log_gifs = logging_config["log_gifs"]
        gif_rollout_steps = logging_config["gif_rollout_steps"] if log_gifs else 0
        log_gif_every_n_rollouts = logging_config["log_gif_every_n_rollouts"] if log_gifs else float('inf')

        callback = WandbCallback(
            project="exploration-marl",
            config=wandb_config,
            gif_rollout_steps=gif_rollout_steps,
            log_gif_every_n_rollouts=log_gif_every_n_rollouts,
        )

    # Train the agent
    print("Starting A2C training...")
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callback
    )

    # Save the trained model
    model.save(config["training"]["save_path"])
    print(f"Training complete. Model saved to {config['training']['save_path']}.zip")