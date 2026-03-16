"""
Simplified base on-policy algorithm for PPO.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

import sys
import time
from collections import deque
from typing import Optional
import numpy as np
import torch as th
from gymnasium import spaces
import gymnasium as gym

from ppo.buffer import RolloutBuffer
from ppo.policy import ActorCriticPolicy
from ppo.utils import Schedule, FloatSchedule, obs_as_tensor, safe_mean, update_learning_rate, get_device, set_random_seed


class SimplePPOLogger:
    """
    Simple logger for PPO training.
    Stores metrics and can be extended to log to different outputs.
    """

    def __init__(self):
        self.name_to_value = {}
        self.name_to_excluded = {}

    def record(self, key: str, value: float, exclude: Optional[str] = None) -> None:
        """Record a key-value pair."""
        self.name_to_value[key] = value
        if exclude:
            self.name_to_excluded[key] = exclude

    def dump(self, step: int = 0) -> None:
        """Dump logged values (override this for custom logging)."""
        pass


class OnPolicyAlgorithm:
    """
    Base class for on-policy algorithms like PPO.

    :param policy: Policy class (ActorCriticPolicy)
    :param env: Training environment
    :param learning_rate: Learning rate or schedule
    :param n_steps: Number of steps to run for each environment per update
    :param gamma: Discount factor
    :param gae_lambda: GAE lambda parameter
    :param ent_coef: Entropy coefficient for the loss
    :param vf_coef: Value function coefficient for the loss
    :param max_grad_norm: Maximum gradient norm for gradient clipping
    :param device: PyTorch device
    :param seed: Random seed
    """

    def __init__(
        self,
        policy: type[ActorCriticPolicy],
        env: gym.Env,
        learning_rate: float | Schedule,
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        device: th.device | str = "cpu",
        seed: Optional[int] = None,
    ):
        self.policy_class = policy
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = get_device(device)
        self.seed = seed

        # Get observation and action spaces from environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = 1  # Simplified for single environment

        # Training state
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._num_timesteps_at_start = 0
        self._n_updates = 0
        self._current_progress_remaining = 1.0
        self._last_obs = None
        self._last_episode_starts = None
        self.start_time = 0.0

        # Episode info buffers
        self.ep_info_buffer = deque(maxlen=100)
        self.ep_success_buffer = deque(maxlen=100)

        # Logger
        self.logger = SimplePPOLogger()

        # Setup will be called by subclass
        self.policy: Optional[ActorCriticPolicy] = None
        self.rollout_buffer: Optional[RolloutBuffer] = None
        self.lr_schedule: Optional[Schedule] = None

    def _setup_lr_schedule(self) -> None:
        """Transform learning rate to schedule."""
        self.lr_schedule = FloatSchedule(self.learning_rate)

    def _setup_model(self) -> None:
        """Create networks and buffers."""
        self._setup_lr_schedule()
        set_random_seed(self.seed)

        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Create policy
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
        )
        self.policy = self.policy.to(self.device)

    def _update_learning_rate(self, optimizer: th.optim.Optimizer) -> None:
        """Update learning rate according to schedule."""
        update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """Update progress remaining."""
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_info_buffer(self, infos: list[dict], dones: np.ndarray) -> None:
        """Update episode info buffer."""
        for idx, info in enumerate(infos):
            if dones[idx]:
                if "episode" in info:
                    ep_info = info["episode"]
                    self.ep_info_buffer.append(ep_info)
                if "is_success" in info:
                    self.ep_success_buffer.append(info["is_success"])

    def collect_rollouts(
        self,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        callback=None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill the rollout buffer.

        :param rollout_buffer: Buffer to fill
        :param n_rollout_steps: Number of experiences to collect
        :param callback: Optional callback (simplified, not used)
        :return: True if collection was successful
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert observation to tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # Add batch dimension if needed
                if len(obs_tensor.shape) == len(self.observation_space.shape):
                    obs_tensor = obs_tensor.unsqueeze(0)

                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            # Clip actions to valid range for Box spaces
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Perform action in environment
            new_obs, rewards, dones, truncated, infos = self.env.step(clipped_actions[0])

            # Convert new format (terminated, truncated) to old format (done)
            done = dones or truncated

            self.num_timesteps += 1

            # Update episode info buffer
            if done:
                # Store episode info if available
                if "episode" in infos:
                    self.ep_info_buffer.append(infos["episode"])

            n_steps += 1

            # Reshape discrete actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            if done and infos.get("TimeLimit.truncated", False):
                terminal_obs = infos.get("terminal_observation", new_obs)
                terminal_obs_tensor = obs_as_tensor(np.array([terminal_obs]), self.device)
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs_tensor)[0]
                rewards = np.array([rewards + self.gamma * terminal_value.cpu().numpy()])
            else:
                rewards = np.array([rewards])

            # Add to buffer
            rollout_buffer.add(
                np.array([self._last_obs]),
                actions,
                rewards,
                np.array([self._last_episode_starts]),
                values,
                log_probs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = done

        # Compute returns and advantages
        with th.no_grad():
            # Add batch dimension if needed
            last_obs_tensor = obs_as_tensor(np.array([new_obs]), self.device)
            last_values = self.policy.predict_values(last_obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=np.array([done]))

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Implemented by subclass (PPO).
        """
        raise NotImplementedError

    def dump_logs(self, iteration: int = 0) -> None:
        """Write log."""
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")

        if len(self.ep_info_buffer) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        callback=None,
    ):
        """
        Train the model.

        :param total_timesteps: Total number of timesteps to train for
        :param log_interval: Log every n iterations
        :param callback: Optional callback (simplified, not used)
        :return: self
        """
        iteration = 0
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Setup model if not already done
        if self.policy is None:
            self._setup_model()

        # Reset environment
        self._last_obs, _ = self.env.reset(seed=self.seed)
        self._last_episode_starts = False

        self.start_time = time.time_ns()

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
                callback=callback,
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Log training info
            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)

            self.train()

        return self

    def set_logger(self, logger) -> None:
        """Set logger."""
        self.logger = logger
