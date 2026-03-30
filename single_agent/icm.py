import torch
import torch.nn as nn
import yaml
import gymnasium as gym

from agent import IMPALA
from utils import state_loss

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
        # Preprocess observations to match SB3's preprocessing:
        # 1. Convert uint8 to float and normalize to [0, 1]
        # 2. Transpose from (batch, H, W, C) to (batch, C, H, W)
        if s.dtype == torch.uint8:
            s = s.float() / 255.0
        # Check if we need to transpose (channels last -> channels first)
        if s.dim() == 4 and s.shape[-1] <= 8:  # Heuristic: small last dim is likely channels
            s = s.permute(0, 3, 1, 2)
        elif s.dim() == 3 and s.shape[-1] <= 8:
            s = s.permute(2, 0, 1)
        return self.encoder(s)


class ICM(nn.Module):

    def __init__(self, config):

        super().__init__()
        
        action_space = config["action_space"]
        input_dims = config["input_dims"]
        layer_norm = config["layer_norm"]

        self.state_encoder = StateEncoder(input_dims)

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(f"ICM currently only supports discrete actions, got {action_space}")

        num_action_logits = action_space.n
        self.num_action_logits = num_action_logits
        latent_state_size = self.state_encoder._latent_size
        # TODO handle continuous actions

        norm_layer = nn.LayerNorm if layer_norm else nn.Identity
        features = lambda x: x if layer_norm else None

        self.action_model = nn.Sequential(
            nn.Linear(2 * latent_state_size, 64),
            #norm_layer(features(64)),
            nn.ReLU(),
            nn.Linear(64, 32),
            #norm_layer(features(32)),
            nn.ReLU(),
            nn.Linear(32, num_action_logits)
        )

        # self.action_encoder = nn.Sequential(
        #     nn.Linear(num_action_logits, latent_action_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(latent_action_size * 2, latent_action_size)
        # )

        self.transition_model = nn.Sequential(
            nn.Linear(latent_state_size + num_action_logits, 64),
            #norm_layer(features(64)),
            nn.ReLU(),
            nn.Linear(64, 32),
            #norm_layer(features(32)),
            nn.ReLU(),
            nn.Linear(32, latent_state_size)
        )

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def _one_hot_action(self, a):
        return torch.nn.functional.one_hot(a, num_classes=self.num_action_logits).float()

    def forward(self, s1, s2, a):
        phi1 = self.state_encoder(s1)
        phi2 = self.state_encoder(s2)
        a_oh = self._one_hot_action(a)

        states = torch.cat([phi1, phi2], dim=-1)
        a_pred = self.action_model(states)

        trans = torch.cat([phi1, a_oh], dim=-1)
        phi_pred = self.transition_model(trans)

        forwards_loss = state_loss(phi_pred, phi2)
        inverse_loss = self.cross_entropy(a_pred, a)

        return phi_pred, a_pred, forwards_loss, inverse_loss


class ICMModule(nn.Module):

    def __init__(self, config, model_id=0):

        super().__init__()
        
        action_space = config["action_space"]
        input_dims = config["input_dims"]
        layer_norm = config["layer_norm"]

        self.state_encoder = StateEncoder(input_dims)
        self.state_encoder.requires_grad_(False)
        self.state_encoder.eval()

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(f"ICM currently only supports discrete actions, got {action_space}")

        num_action_logits = action_space.n
        self.num_action_logits = num_action_logits
        latent_state_size = self.state_encoder._latent_size
        # TODO handle continuous actions

        norm_layer = nn.LayerNorm if layer_norm else nn.Identity
        features = lambda x: x if layer_norm else None

        self.transition_model = nn.Sequential(
            nn.Linear(latent_state_size + num_action_logits, 64),
            #batch_norm_layer(),
            nn.ReLU(),
            nn.Linear(64, 32),
            #batch_norm_layer(),
            nn.ReLU(),
            nn.Linear(32, latent_state_size)
        )

    def _one_hot_action(self, a):
        return torch.nn.functional.one_hot(a, num_classes=self.num_action_logits).float()

    def _embed_state(self, s):
        with torch.no_grad():
            phi = self.state_encoder(s)
        return phi

    def forward(self, s, a):
        with torch.no_grad():
            phi = self.state_encoder(s)
        a_oh = self._one_hot_action(a)

        trans = torch.cat([phi, a_oh], dim=-1)
        phi_pred = self.transition_model(trans)

        return phi_pred
