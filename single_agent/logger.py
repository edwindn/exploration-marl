import numpy as np
from io import BytesIO
from PIL import Image

from typing import Optional, List, Callable
from stable_baselines3.common.callbacks import BaseCallback


class GifLoggingCallback:
    """
    Standalone callback for logging GIF visualizations during training.
    Can be used with both custom PPO implementations and stable-baselines3.

    :param env: The environment to render
    :param gif_rollout_steps: Number of steps to collect for each GIF
    :param log_gif_every_n_rollouts: Log GIF every N rollouts
    """

    def __init__(
        self,
        env,
        gif_rollout_steps: int = 25,
        log_gif_every_n_rollouts: int = 1,
    ):
        self.env = env
        self.gif_rollout_steps = gif_rollout_steps
        self.log_gif_every_n_rollouts = log_gif_every_n_rollouts
        self.rollout_count = 0

    def should_log_gif(self) -> bool:
        """Check if we should log a GIF this rollout."""
        self.rollout_count += 1
        return self.rollout_count % self.log_gif_every_n_rollouts == 0

    def collect_inference_rollout(self, agent, device) -> List[np.ndarray]:
        """
        Collect a short inference rollout using the current policy.

        :param agent: The agent/model to use for inference
        :param device: The torch device (cpu/cuda)
        :return: List of frames (numpy arrays) from the rollout
        """
        import torch

        # Unwrap environment if wrapped in VecEnv
        env = self.env
        if hasattr(env, 'envs'):
            env = env.envs[0]

        # Unwrap to get the actual base environment with _render_frame
        while hasattr(env, 'env'):
            env = env.env

        frames = []
        obs, _ = self.env.reset() if hasattr(self.env, 'reset') else (self.env.envs[0].reset(), None)

        # Handle VecEnv output format
        if isinstance(obs, tuple):
            obs = obs[0]

        # Collect frames for the specified number of steps
        for _ in range(self.gif_rollout_steps):
            # Render the current state
            frame = env._render_frame()
            frames.append(frame)

            # Get action from current policy (deterministic for consistency)
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()

            # Extract scalar from array if needed
            if isinstance(action, np.ndarray):
                if action.ndim > 0:
                    action = action.flatten()

            # Take step in environment
            step_result = self.env.step(action)
            if hasattr(self.env, 'envs'):  # VecEnv
                obs, reward, terminated, truncated, info = step_result
                terminated = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                truncated = truncated[0] if isinstance(truncated, np.ndarray) else truncated
            else:
                obs, reward, terminated, truncated, info = step_result

            # If episode ends, reset
            if terminated or truncated:
                reset_result = self.env.reset() if hasattr(self.env, 'reset') else self.env.envs[0].reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        return frames

    def create_gif_from_frames(self, frames: List[np.ndarray]):
        """
        Create a WandB-compatible GIF from a list of frames.

        :param frames: List of numpy arrays representing frames
        :return: wandb.Video object for logging
        """
        import wandb

        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]

        # Save as GIF to in-memory buffer
        buffer = BytesIO()
        pil_frames[0].save(
            buffer,
            format='GIF',
            append_images=pil_frames[1:],
            save_all=True,
            duration=100,  # milliseconds per frame
            loop=0  # infinite loop
        )
        buffer.seek(0)

        # Create wandb Video object from the GIF
        return wandb.Video(buffer, fps=10, format="gif")


class WandbCallback(BaseCallback):
    """
    Callback that logs training metrics to Weights & Biases.

    :param project: W&B project name
    :param config: Dictionary of hyperparameters to log
    :param include_metrics: List of metric prefixes to include (e.g., ["train/", "rollout/"])
        If None, all metrics are logged
    :param exclude_metrics: List of specific metrics to exclude (e.g., ["train/n_updates"])
    :param metric_filter: Custom filter function that takes (key, value) and returns bool
    :param verbose: Verbosity level
    """

    def __init__(
        self,
        project: str = "exploration-marl",
        config: dict = None,
        include_metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
        metric_filter: Optional[Callable[[str, any], bool]] = None,
        gif_rollout_steps: int = 25,
        log_gif_every_n_rollouts: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.project = project
        self.config = config or {}
        self.include_metrics = include_metrics
        self.exclude_metrics = exclude_metrics or []
        self.metric_filter = metric_filter
        self.gif_rollout_steps = gif_rollout_steps
        self.log_gif_every_n_rollouts = log_gif_every_n_rollouts
        self.wandb = None

        self.episode_rewards = []
        self.episode_visitation_percentages = []
        self.rollout_count = 0

    def _on_training_start(self) -> None:
        import wandb
        self.wandb = wandb

        wandb.init(
            project=self.project,
            config=self.config,
        )

    def _on_step(self) -> bool:
        """
        This method is called after each env.step().
        Track episode rewards from info dict returned by env.step.
        """
        assert self.training_env.num_envs == 1

        infos = self.locals["infos"]

        assert len(infos) == 1, "Cannot use more than one parallel env"
        info = infos[0]

        self.episode_rewards.append(info["reward"])
        if "visitation_percentage" in info:
            self.episode_visitation_percentages.append(info["visitation_percentage"])

        return True

    def _collect_inference_rollout(self) -> List[np.ndarray]:
        """
        Collect a short inference rollout using the current policy.
        Creates a fresh environment to avoid interfering with training state.

        :return: List of frames (numpy arrays) from the rollout
        """
        from environment.env import NavEnv

        # Create a fresh environment instance for GIF collection
        gif_env = NavEnv()

        frames = []
        obs, _ = gif_env.reset()

        # Collect frames for the specified number of steps
        for _ in range(self.gif_rollout_steps):
            # Render the current state
            frame = gif_env._render_frame()
            frames.append(frame)

            # Get action from current policy (deterministic for consistency)
            action, _ = self.model.predict(obs, deterministic=True)

            # Extract scalar from array if needed (for discrete action spaces)
            if isinstance(action, np.ndarray):
                action = action.item()

            # Take step in environment
            obs, reward, terminated, truncated, info = gif_env.step(action)

            # If episode ends, we still continue to show full 25 steps
            if terminated or truncated:
                obs, _ = gif_env.reset()

        # Clean up the temporary environment
        gif_env.close()
        del gif_env

        return frames

    def _create_gif_from_frames(self, frames: List[np.ndarray]):
        """
        Create a WandB-compatible GIF from a list of frames.

        :param frames: List of numpy arrays representing frames
        :return: wandb.Video object for logging
        """
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]

        # Save as GIF to in-memory buffer
        buffer = BytesIO()
        pil_frames[0].save(
            buffer,
            format='GIF',
            append_images=pil_frames[1:],
            save_all=True,
            duration=100,  # milliseconds per frame
            loop=0  # infinite loop
        )
        buffer.seek(0)

        # Create wandb Video object from the GIF
        return self.wandb.Video(buffer, fps=10, format="gif")

    def _should_log_metric(self, key: str, value: any) -> bool:
        """
        Determine if a metric should be logged based on filters.

        :param key: Metric name
        :param value: Metric value
        :return: True if metric should be logged
        """
        # Skip non-numeric values
        if not isinstance(value, (int, float)):
            return False

        # Check exclude list (specific keys)
        if key in self.exclude_metrics:
            return False

        # Check include list (prefixes)
        if self.include_metrics is not None:
            if not any(key.startswith(prefix) for prefix in self.include_metrics):
                return False

        # Apply custom filter if provided
        if self.metric_filter is not None:
            if not self.metric_filter(key, value):
                return False

        return True

    def _on_rollout_end(self) -> None:
        """Log metrics after each rollout."""
        if self.wandb is None:
            return

        # Get all metrics from the SB3 logger and filter them
        metrics = {}
        for key, value in self.logger.name_to_value.items():
            if self._should_log_metric(key, value):
                metrics[key] = value

        # Add custom mean reward from episodes since last logging
        assert self.episode_rewards
        metrics["train/mean_extrinsic_reward"] = np.mean(self.episode_rewards)

        # Add visitation percentage if available
        if self.episode_visitation_percentages:
            metrics["train/mean_visitation_percentage"] = np.mean(self.episode_visitation_percentages)

        # Clear the rewards and visitation percentages for next logging interval
        self.episode_rewards = []
        self.episode_visitation_percentages = []

        # Add timestep information
        metrics["timesteps"] = self.num_timesteps

        # Collect and log GIF visualization at specified intervals
        self.rollout_count += 1
        if self.rollout_count % self.log_gif_every_n_rollouts == 0:
            frames = self._collect_inference_rollout()
            gif = self._create_gif_from_frames(frames)
            metrics["rollout/visualization"] = gif

        # Log to wandb
        if metrics:
            self.wandb.log(metrics, step=self.num_timesteps)

    def _on_training_end(self) -> None:
        """Close wandb run at the end of training."""
        if self.wandb is not None:
            self.wandb.finish()
            if self.verbose > 0:
                print("Wandb run finished.")
