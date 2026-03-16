"""
Simplified neural network modules for PPO.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

from typing import Union
import torch as th
from torch import nn
from ppo.utils import get_device


class MlpExtractor(nn.Module):
    """
    MLP network that outputs separate latent representations for policy and value networks.

    :param feature_dim: Dimension of the input features
    :param net_arch: Architecture specification (list or dict with 'pi' and 'vf' keys)
    :param activation_fn: Activation function to use
    :param device: PyTorch device
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__()
        device = get_device(device)

        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # Parse network architecture
        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # Build policy network
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim

        # Build value network
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dimensions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forward pass through both networks.

        :param features: Input features
        :return: Policy latent representation, value latent representation
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through policy network."""
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through value network."""
        return self.value_net(features)
