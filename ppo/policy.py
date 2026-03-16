"""
Simplified ActorCritic policy for PPO.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

from functools import partial
from typing import Optional, Union
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN

from ppo.distributions import Distribution, make_proba_distribution
from ppo.networks import MlpExtractor
from ppo.preprocessing import preprocess_obs
from ppo.utils import Schedule


class ActorCriticPolicy(nn.Module):
    """
    Policy class for actor-critic algorithms (PPO, A2C, etc.).
    Combines an actor (policy) and a critic (value function).

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: Architecture of policy and value networks
    :param activation_fn: Activation function
    :param ortho_init: Whether to use orthogonal initialization
    :param features_extractor_class: Features extractor class
    :param features_extractor_kwargs: Kwargs for features extractor
    :param normalize_images: Whether to normalize images
    :param optimizer_class: Optimizer class
    :param optimizer_kwargs: Additional optimizer kwargs
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}
        self.normalize_images = normalize_images
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_schedule = lr_schedule

        # Network architecture
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # Create features extractor
        self.features_extractor = features_extractor_class(
            observation_space, **self.features_extractor_kwargs
        )
        self.features_dim = self.features_extractor.features_dim

        # Create action distribution
        self.action_dist = make_proba_distribution(action_space)

        # Build networks
        self._build(lr_schedule)

    @property
    def device(self) -> th.device:
        """Infer device from parameters."""
        for param in self.parameters():
            return param.device
        return th.device("cpu")

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and optimizer.

        :param lr_schedule: Learning rate schedule
        """
        # Create MLP extractor
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Create action network based on action distribution type
        from ppo.distributions import DiagGaussianDistribution, CategoricalDistribution
        from ppo.distributions import MultiCategoricalDistribution, BernoulliDistribution

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=0.0
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Create value network
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Orthogonal initialization
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Create optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1.0) -> None:
        """
        Orthogonal initialization.

        :param module: Module to initialize
        :param gain: Gain for orthogonal initialization
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess and extract features from observations.

        :param obs: Observations
        :return: Extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all networks (actor and critic).

        :param obs: Observation
        :param deterministic: Whether to use deterministic actions
        :return: Actions, values, log probabilities
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get value
        values = self.value_net(latent_vf)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Get action distribution from latent codes.

        :param latent_pi: Latent code for actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        from ppo.distributions import DiagGaussianDistribution, CategoricalDistribution
        from ppo.distributions import MultiCategoricalDistribution, BernoulliDistribution

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy.

        :param obs: Observations
        :param actions: Actions
        :return: Values, log probabilities, entropy
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get value estimates.

        :param obs: Observations
        :return: Value estimates
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def set_training_mode(self, mode: bool) -> None:
        """
        Set training mode.

        :param mode: True for training mode, False for eval mode
        """
        self.train(mode)


class ActorCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms.
    Uses NatureCNN as the default features extractor.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param args: Positional arguments passed to ActorCriticPolicy
    :param kwargs: Keyword arguments passed to ActorCriticPolicy
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args,
        **kwargs,
    ):
        # Use NatureCNN as default features extractor for CNN policy
        kwargs["features_extractor_class"] = kwargs.get("features_extractor_class", NatureCNN)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
