"""
Simplified RolloutBuffer for PPO algorithm.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

from typing import Generator, NamedTuple, Optional
import numpy as np
import torch as th
from gymnasium import spaces


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms like PPO.

    :param buffer_size: Max number of elements in the buffer (n_steps)
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for GAE (Generalized Advantage Estimator)
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments (default: 1)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = th.device(device) if isinstance(device, str) else device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs

        # Get observation and action dimensions
        self.obs_shape = observation_space.shape
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = 1
        elif isinstance(action_space, spaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(len(action_space.nvec))
        elif isinstance(action_space, spaces.MultiBinary):
            self.action_dim = int(action_space.n)
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        self.pos = 0
        self.full = False
        self.generator_ready = False

        self.reset()

    def reset(self) -> None:
        """Reset the buffer."""
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        Add a new transition to the buffer.

        :param obs: Observation
        :param action: Action
        :param reward: Reward
        :param episode_start: Start of episode signal
        :param value: Estimated value of the current state
        :param log_prob: Log probability of the action
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Compute the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

        :param last_values: State value estimation for the last step
        :param dones: If the last step was a terminal step
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # TD(lambda) estimator
        self.returns = self.advantages + self.values

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and flatten axes 0 (buffer_size) and 1 (n_envs).

        Converts shape from [n_steps, n_envs, ...] to [n_steps * n_envs, ...]
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.

        :param array: Numpy array
        :param copy: Whether to copy the data
        :return: PyTorch tensor
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """
        Get samples from the buffer in mini-batches.

        :param batch_size: Mini-batch size (if None, return full buffer)
        :return: Generator yielding RolloutBufferSamples
        """
        assert self.full, "Buffer must be full before sampling"

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything if no batch size specified
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        """
        Get specific samples from the buffer.

        :param batch_inds: Indices of samples to return
        :return: RolloutBufferSamples
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
