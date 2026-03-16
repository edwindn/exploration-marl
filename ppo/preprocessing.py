"""
Simplified preprocessing utilities for PPO.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

from typing import Union
import torch as th
from gymnasium import spaces
from torch.nn import functional as F


def is_image_space(observation_space: spaces.Space) -> bool:
    """
    Check if observation space is a valid image (3D Box space).

    :param observation_space: Observation space
    :return: True if it's an image space
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        return True
    return False


def preprocess_obs(
    obs: Union[th.Tensor, dict[str, th.Tensor]],
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, dict[str, th.Tensor]]:
    """
    Preprocess observation for neural network input.
    Normalizes images by dividing by 255 and creates one-hot vectors for discrete obs.

    :param obs: Observation tensor or dict of tensors
    :param observation_space: Observation space
    :param normalize_images: Whether to normalize images
    :return: Preprocessed observation
    """
    if isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    assert isinstance(obs, th.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding
        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Concatenation of one-hot encodings
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
