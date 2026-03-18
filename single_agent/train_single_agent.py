import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#from stable_baselines3.ppo import PPO
from single_agent.ppo_wrapper import PPO_ICM as PPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import gymnasium as gym

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
    parser = argparse.ArgumentParser(description="Train PPO agent on navigation environment")
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

    # Initialize PPO with custom implementation from ppo.py
    ppo_config = config["ppo"]
    model = PPO(
        train_icm=ppo_config["train_icm"],
        icm_learning_rate=ppo_config["icm_learning_rate"],
        eta=ppo_config["eta"],
        beta=ppo_config["beta"],
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=ppo_config["num_train_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        learning_rate=ppo_config["learning_rate"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
        verbose=ppo_config["verbose"],
    )

    # Configure logger to disable stdout table output
    # Only keep csv and tensorboard if needed, remove 'stdout'
    new_logger = configure(folder=None, format_strings=["csv"])
    model.set_logger(new_logger)

    # Create wandb callback with all config for logging (if enabled)
    callback = None
    if not args.disable_wandb:
        wandb_config = {
            "algorithm": "PPO",
            **env_config,
            **config["ppo"],
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
    print("Starting PPO training...")
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callback
    )

    # Save the trained model
    model.save(config["training"]["save_path"])
    print(f"Training complete. Model saved to {config['training']['save_path']}.zip")