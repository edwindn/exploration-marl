import numpy as np
from io import BytesIO
from PIL import Image

from typing import Optional, List, Callable
from stable_baselines3.common.callbacks import BaseCallback

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

        return True

    def _collect_inference_rollout(self) -> List[np.ndarray]:
        """
        Collect a short inference rollout using the current policy.

        :return: List of frames (numpy arrays) from the rollout
        """
        # Get the unwrapped environment (SB3 wraps in VecEnv and Monitor)
        env = self.training_env.envs[0]
        # Unwrap to get the actual NavEnv instance
        while hasattr(env, 'env'):
            env = env.env

        frames = []
        obs, _ = self.training_env.envs[0].reset()

        # Collect frames for the specified number of steps
        for _ in range(self.gif_rollout_steps):
            # Render the current state
            frame = env._render_frame()
            frames.append(frame)

            # Get action from current policy (deterministic for consistency)
            action, _ = self.model.predict(obs, deterministic=True)

            # Extract scalar from array if needed (for discrete action spaces)
            if isinstance(action, np.ndarray):
                action = action.item()

            # Take step in environment
            obs, reward, terminated, truncated, info = self.training_env.envs[0].step(action)

            # If episode ends, we still continue to show full 25 steps
            if terminated or truncated:
                obs, _ = self.training_env.envs[0].reset()

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
        metrics["train/mean_reward"] = np.mean(self.episode_rewards)
        # Clear the rewards for next logging interval
        self.episode_rewards = []

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
