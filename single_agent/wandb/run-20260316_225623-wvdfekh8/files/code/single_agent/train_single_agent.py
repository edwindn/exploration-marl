import argparse
import yaml
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import NavEnv
from single_agent.ppo import train


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

    # Get PPO config
    ppo_config = config["ppo"]
    ppo_config["total_timesteps"] = config["training"]["total_timesteps"]

    # Get IMPALA encoder config
    impala_config = config["agent"]["backbone"]

    # Prepare wandb config (if enabled)
    use_wandb = not args.disable_wandb
    wandb_project = config["logging"]["wandb_project"] if use_wandb else None
    wandb_config = None
    if use_wandb:
        wandb_config = {
            "algorithm": "PPO",
            **env_config,
            **config["ppo"],
            **config["agent"]["policy"],
            **config["training"],
        }

    # Train the agent using CleanRL PPO
    print("Starting PPO training...")
    train(
        env=env,
        config=ppo_config,
        impala_config=impala_config,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
    )

    print(f"Training complete!")
