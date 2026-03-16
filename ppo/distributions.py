"""
Simplified probability distributions for PPO.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical, Normal
import numpy as np


class Distribution(ABC):
    """Abstract base class for action distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs):
        """Create the layers and parameters that represent the distribution."""
        pass

    @abstractmethod
    def proba_distribution(self, *args, **kwargs):
        """Set parameters of the distribution."""
        pass

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """Returns the log likelihood of actions."""
        pass

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """Returns Shannon's entropy of the probability."""
        pass

    @abstractmethod
    def sample(self) -> th.Tensor:
        """Returns a sample from the probability distribution."""
        pass

    @abstractmethod
    def mode(self) -> th.Tensor:
        """Returns the most likely action (deterministic output)."""
        pass

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic: Whether to use deterministic or stochastic actions
        :return: Actions
        """
        if deterministic:
            return self.mode()
        return self.sample()


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Sum components of the log_prob or entropy for independent actions.

    :param tensor: Tensor of shape (n_batch, n_actions) or (n_batch,)
    :return: Tensor of shape (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance for continuous actions.

    :param action_dim: Dimension of the action space
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        """
        Create the layers and parameter for the Gaussian distribution.

        :param latent_dim: Dimension of the last layer of the policy
        :param log_std_init: Initial value for the log standard deviation
        :return: Mean actions layer and log std parameter
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        """
        Create the distribution given its parameters.

        :param mean_actions: Mean actions
        :param log_std: Log standard deviation
        :return: self
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Get the log probabilities of actions."""
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        """Get the entropy of the distribution."""
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        """Sample from the distribution (reparametrization trick)."""
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """Get the mean (most likely action)."""
        return self.distribution.mean


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer for the categorical distribution.

        :param latent_dim: Dimension of the last layer of the policy network
        :return: Action logits layer
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor):
        """
        Create the distribution from action logits.

        :param action_logits: Logits for the categorical distribution
        :return: self
        """
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Get the log probabilities of actions."""
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        """Get the entropy of the distribution."""
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        """Sample from the distribution."""
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        """Get the most likely action."""
        return th.argmax(self.distribution.probs, dim=1)


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi-discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: list[int]):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer for the multi-categorical distribution.

        :param latent_dim: Dimension of the last layer of the policy network
        :return: Action logits layer
        """
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor):
        """
        Create the distribution from action logits.

        :param action_logits: Logits for the multi-categorical distribution
        :return: self
        """
        self.distribution = [
            Categorical(logits=split) for split in th.split(action_logits, list(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Get the log probabilities of actions."""
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        """Get the entropy of the distribution."""
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        """Sample from the distribution."""
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        """Get the most likely actions."""
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for multi-binary actions.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer for the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer of the policy network
        :return: Action logits layer
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor):
        """
        Create the distribution from action logits.

        :param action_logits: Logits for the Bernoulli distribution
        :return: self
        """
        self.distribution = th.distributions.Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Get the log probabilities of actions."""
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> th.Tensor:
        """Get the entropy of the distribution."""
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        """Sample from the distribution."""
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        """Get the most likely actions."""
        return th.round(self.distribution.probs)


def make_proba_distribution(action_space: spaces.Space) -> Distribution:
    """
    Create an appropriate distribution for the given action space.

    :param action_space: Action space
    :return: Distribution instance
    """
    if isinstance(action_space, spaces.Box):
        action_dim = int(np.prod(action_space.shape))
        return DiagGaussianDistribution(action_dim)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(int(action_space.n))
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(action_space.n, int), \
            f"Multi-dimensional MultiBinary({action_space.n}) not supported"
        return BernoulliDistribution(action_space.n)
    else:
        raise NotImplementedError(
            f"Probability distribution not implemented for action space {type(action_space)}"
        )
