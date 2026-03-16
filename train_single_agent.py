import argparse
import torch
import torch.nn as nn
import yaml
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import gymnasium as gym

from env import NavEnv
from agent import IMPALA
from ppo import PPO
from logger import WandbCallback


class IMPALAExtractor(BaseFeaturesExtractor):
    """Wraps the IMPALA CNN from agent.py as an SB3 feature extractor."""

    def __init__(self, observation_space, config_path: str = "train_config.yaml"):
        # Load config to get feature_size
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        features_dim = train_config['backbone']['feature_size']

        super().__init__(observation_space, features_dim)
        self.impala = IMPALA(config_path)

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
    with open("train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("env_config.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    # Create environment
    env = NavEnv()
    check_env(env, warn=True)

    # Configure policy to use IMPALA feature extractor
    policy_kwargs = dict(
        features_extractor_class=IMPALAExtractor,
        net_arch=config["policy"]["net_arch"],
        activation_fn=nn.ReLU,
    )

    # Initialize PPO with custom implementation from ppo.py
    ppo_config = config["ppo"]
    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=ppo_config["n_steps"],
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
            **config["policy"],
            **config["training"],
        }

        # Get GIF logging config
        logging_config = config["logging"]
        log_gifs = logging_config["log_gifs"]
        gif_rollout_steps = logging_config["gif_rollout_steps"] if log_gifs else 0
        log_gif_every_n_rollouts = logging_config["log_gif_every_n_rollouts"] if log_gifs else float('inf')

        callback = WandbCallback(
            project=logging_config["wandb_project"],
            config=wandb_config,
            gif_rollout_steps=gif_rollout_steps,
            log_gif_every_n_rollouts=log_gif_every_n_rollouts,
            verbose=logging_config["verbose"],
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
