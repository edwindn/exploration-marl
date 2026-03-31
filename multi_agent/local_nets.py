import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize

    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, config, discrete_actions=False, action_dim=None, action_low=None, action_high=None, device='cpu'):
        super().__init__()
        self.config = config
        self.discrete_actions = discrete_actions
        self.action_dim = action_dim if action_dim is not None else config.action_dim

        # Input size is latent_dim from config
        input_size = config.latent_dim

        # For continuous actions, output is 2*action_dim (mean and log_std)
        # For discrete actions, output is num_actions (logits for each possible action)
        if discrete_actions:
            output_size = config.num_actions
        else:
            output_size = 2 * self.action_dim

        self.network = sequentialModel1D(
            input_size,
            [config.mlp_dim] * 2,  # 2 hidden layers with mlp_dim units
            output_size,
            "Tanh"
        )

        # Only needed for continuous actions
        if not discrete_actions:
            action_low = [action_low if action_low is not None else -1.0] * self.action_dim
            action_high = [action_high if action_high is not None else 1.0] * self.action_dim
            self.register_buffer("actionScale", ((torch.tensor(action_high, device=device) - torch.tensor(action_low, device=device)) / 2.0))
            self.register_buffer("actionBias", ((torch.tensor(action_high, device=device) + torch.tensor(action_low, device=device)) / 2.0))

    def forward(self, x, task=None, training=False):
        if task is not None:
            raise NotImplementedError("Task choice in actor not yet implemented")

        if self.discrete_actions:
            # print('actor: ', x.shape)
            logits = self.network(x)
            distribution = Categorical(logits=logits)
            return distribution
            # action = distribution.sample()

            # if training:
            #     logprobs = distribution.log_prob(action)
            #     entropy = distribution.entropy()
            #     return action, logprobs, entropy
            # else:
            #     return action
        else:
            # Continuous actions
            logStdMin, logStdMax = -5, 2
            mean, logStd = self.network(x).chunk(2, dim=-1)
            logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1)  # (-1, 1) to (min, max)
            std = torch.exp(logStd)

            distribution = Normal(mean, std)
            return distribution
            # sample = distribution.sample()
            # sampleTanh = torch.tanh(sample)
            # action = sampleTanh * self.actionScale + self.actionBias

            # if training:
            #     logprobs = distribution.log_prob(sample)
            #     logprobs -= torch.log(self.actionScale * (1 - sampleTanh.pow(2)) + 1e-6)
            #     entropy = distribution.entropy()
            #     return action, logprobs.sum(-1), entropy.sum(-1)
            # else:
            #     return action


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input size is latent_dim from config
        input_size = config.latent_dim

        self.network = sequentialModel1D(
            input_size,
            [config.mlp_dim] * 2,  # 2 hidden layers with mlp_dim units
            2,  # Output mean and log_std
            "Tanh"
        )

    def forward(self, x, task=None, deterministic=True):
        if task is not None:
            raise NotImplementedError("Task choice in critic not yet implemented")
        # print('critic: ', x.shape)
        mean, logStd = self.network(x).chunk(2, dim=-1)
        if deterministic:
            return mean
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
