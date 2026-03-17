import torch
import torch.nn as nn
import yaml
import gymnasium as gym

from agent import IMPALA


class StateEncoder(nn.Module):

    def __init__(self, input_dims, config_path: str = "train_config.yaml"):
        super().__init__()

        # Load ICM config from train_config.yaml
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        icm_config = train_config['agent']['icm']

        self.encoder = IMPALA(input_dims, icm_config)
        self._latent_size = self.encoder._output_dims[0]

    def forward(self, s):
        return self.encoder(s)


class ICM(nn.Module):

    def __init__(self, action_space, input_dims):

        super().__init__()
        
        self.state_encoder = StateEncoder(input_dims)

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(f"ICM currently only supports discrete actions, got {action_space}")

        num_action_logits = action_space.n
        self.num_action_logits = num_action_logits
        latent_action_size = 8
        latent_state_size = self.state_encoder._latent_size
        # TODO handle continuous actions

        self.action_model = nn.Sequential(
            nn.Linear(2 * latent_state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_action_logits)
        )

        # self.action_encoder = nn.Sequential(
        #     nn.Linear(num_action_logits, latent_action_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(latent_action_size * 2, latent_action_size)
        # )

        self.transition_model = nn.Sequential(
            nn.Linear(latent_state_size + latent_action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_state_size)
        )

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def _one_hot_action(a):
        return torch.nn.functional.one_hot(a, num_classes=self.num_action_logits).float()

    def forward(self, s1, s2, a):
        phi1 = self.state_encoder(s1)
        phi2 = self.state_encoder(s2)
        a_oh = self._one_hot_action(a)

        states = torch.cat([phi1, phi2], dim=-1)
        a_pred = self.action_model(states)

        trans = torch.cat([phi1, a_oh], dim=-1)
        phi_pred = self.transition_model(trans)

        forwards_loss = 0.5 * torch.sum((phi2 - phi_pred) ** 2)
        inverse_loss = self.cross_entropy(a_pred, a)

        return phi_pred, a_pred, forwards_loss, inverse_loss
