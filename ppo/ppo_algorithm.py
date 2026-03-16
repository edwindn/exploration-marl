"""
Simplified PPO algorithm implementation.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

import warnings
from typing import Optional
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import gymnasium as gym

from ppo.base_algorithm import OnPolicyAlgorithm
from ppo.policy import ActorCriticPolicy, ActorCriticCnnPolicy
from ppo.utils import Schedule, FloatSchedule, explained_variance


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) with clipping.

    Paper: https://arxiv.org/abs/1707.06347
    Code based on OpenAI Spinning Up and Stable Baselines3.

    :param policy: Policy type ("MlpPolicy" or "CnnPolicy")
    :param env: The environment to learn from
    :param learning_rate: Learning rate (can be a schedule)
    :param n_steps: Number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epochs when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: GAE lambda parameter
    :param clip_range: Clipping parameter for the policy (can be a schedule)
    :param clip_range_vf: Clipping parameter for the value function (optional)
    :param normalize_advantage: Whether to normalize advantages
    :param ent_coef: Entropy coefficient for the loss
    :param vf_coef: Value function coefficient for the loss
    :param max_grad_norm: Maximum value for gradient clipping
    :param target_kl: Target KL divergence for early stopping (optional)
    :param policy_kwargs: Additional policy kwargs
    :param verbose: Verbosity level
    :param device: PyTorch device
    :param seed: Random seed
    """

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: gym.Env,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: Optional[float | Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        policy_kwargs: Optional[dict] = None,
        verbose: int = 0,
        device: th.device | str = "auto",
        seed: Optional[int] = None,
    ):
        # Parse policy string to class
        if isinstance(policy, str):
            if policy == "MlpPolicy":
                policy_class = ActorCriticPolicy
            elif policy == "CnnPolicy":
                policy_class = ActorCriticCnnPolicy
            else:
                raise ValueError(f"Unknown policy: {policy}")
        else:
            policy_class = policy

        # Add policy_kwargs to policy_class
        if policy_kwargs is not None:
            original_init = policy_class.__init__

            def new_init(self, *args, **kwargs):
                kwargs.update(policy_kwargs)
                original_init(self, *args, **kwargs)

            policy_class.__init__ = new_init

        super().__init__(
            policy=policy_class,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            seed=seed,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.verbose = verbose

        # Sanity checks
        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1 for advantage normalization"

        # Check that n_steps * n_envs > 1
        buffer_size = self.n_envs * self.n_steps
        assert buffer_size > 1 or not normalize_advantage, \
            f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.n_envs}"

        # Warn if buffer_size is not a multiple of batch_size
        untruncated_batches = buffer_size // batch_size
        if buffer_size % batch_size > 0:
            warnings.warn(
                f"You have specified a mini-batch size of {batch_size}, "
                f"but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`, "
                f"after every {untruncated_batches} untruncated mini-batches, "
                f"there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                f"Info: (n_steps={self.n_steps} and n_envs={self.n_envs})"
            )

    def _setup_model(self) -> None:
        """Setup model, including clip range schedules."""
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive"
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        # Optional: clip range for value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Evaluate actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    # Approximate entropy when no analytical form is available
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1

            if not continue_training:
                break

        # Compute explained variance
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )

        # Log metrics
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

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
        :param callback: Optional callback
        :return: self
        """
        return super().learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,
        )

    def save(self, path: str) -> None:
        """
        Save the model.

        :param path: Path to save the model (without .zip extension)
        """
        save_path = path if path.endswith(".zip") else f"{path}.zip"
        th.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
        }, save_path)
        if self.verbose >= 1:
            print(f"Model saved to {save_path}")
