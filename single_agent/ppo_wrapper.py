import warnings
from typing import Any, ClassVar, TypeVar, Optional, Generator, NamedTuple
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from itertools import chain

from stable_baselines3.ppo import PPO
from stable_baselines3.common.utils import explained_variance, obs_as_tensor, get_schedule_fn, update_learning_rate
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from single_agent.icm import ICM


class RolloutBufferSamplesWithNextObs(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    next_observations: th.Tensor


class RolloutBufferWithNextObs(RolloutBuffer):
    """Extended RolloutBuffer that also stores next observations for ICM training."""

    next_observations: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        next_obs: Optional[np.ndarray] = None,
    ) -> None:
        """Add transition to buffer, including next observation."""
        super().add(obs, action, reward, episode_start, value, log_prob)

        # Store next observation at the previous position (we just incremented self.pos)
        if next_obs is not None:
            prev_pos = (self.pos - 1) % self.buffer_size
            if isinstance(self.observation_space, spaces.Discrete):
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            self.next_observations[prev_pos] = np.array(next_obs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "next_observations",  # Added
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamplesWithNextObs:
        # Get parent samples
        parent_samples = super()._get_samples(batch_inds, env)

        # Add next_observations to the samples
        next_obs = self.to_torch(self.next_observations[batch_inds])

        # Return extended samples with next_observations
        return RolloutBufferSamplesWithNextObs(
            observations=parent_samples.observations,
            actions=parent_samples.actions,
            old_values=parent_samples.old_values,
            old_log_prob=parent_samples.old_log_prob,
            advantages=parent_samples.advantages,
            returns=parent_samples.returns,
            next_observations=next_obs,
        )


class PPO_ICM(PPO):

    def __init__(self, train_icm, eta, beta, _lambda, *args, **kwargs):

        env = kwargs.get("env", None)
        assert env is not None, "must provide env to PPO module"

        inp = env.observation_space.shape

        if train_icm:
            self.icm = ICM(
                action_space=env.action_space,
                #input_dims=(inp[2], inp[0], inp[1]),
                input_dims=inp
            )

            self.eta = eta  # unsupervised reward weighting
            self.beta = beta  # forward vs inverse loss weighting
            self._lambda = _lambda

            self.last_mean_forward_loss = None
            self.normalize_intrinsic_reward = False
        else:
            self.icm = None

        super().__init__(*args, **kwargs)

        if self.icm is not None:
            self.icm.to(self.device)

    def _setup_model(self) -> None:
        # Set rollout buffer class before calling super()._setup_model()
        # Parent only sets it if it's None (line 119 in on_policy_algorithm.py)
        if self.icm is not None:
            self.rollout_buffer_class = RolloutBufferWithNextObs

        # Call parent _setup_model which creates policy, rollout buffer, and policy.optimizer
        super()._setup_model()

        # Now create joint optimizer if using ICM (policy now exists)
        if self.icm is not None:
            # Combine ICM and policy parameters
            joint_params = chain(self.icm.parameters(), self.policy.parameters())

            # Create joint optimizer using same settings as policy optimizer
            self.joint_optimizer = self.policy.optimizer_class(
                joint_params,
                lr=self.lr_schedule(1),  # Initial learning rate
                **self.policy.optimizer_kwargs
            )


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # Track ICM metrics for the rollout
        icm_forward_losses = []
        icm_inverse_losses = []
        total_rewards = []  # Track total rewards (extrinsic + intrinsic)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # add intrinsic reward
            if self.icm is not None:
                new_obs_tensor = obs_as_tensor(new_obs, self.device)
                actions_tensor = th.tensor(actions).to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    actions_tensor = actions_tensor.long().flatten()

                # Compute ICM losses for logging (without gradients)
                with th.no_grad():
                    _, _, forward_loss, inverse_loss = self.icm(obs_tensor, new_obs_tensor, actions_tensor)
                    icm_forward_losses.append(forward_loss.cpu().item())
                    icm_inverse_losses.append(inverse_loss.cpu().item())

                if rewards.shape[0] != 1:
                    raise ValueError("Cannot  use parallel environments with current intrinsic reward computation (need to modify to per-env reward)")
                
                intrinsic_reward = forward_loss.cpu().numpy()
                if self.last_mean_forward_loss is not None and self.normalize_intrinsic_reward:
                    intrinsic_reward /= (self.last_mean_forward_loss + 1.0e-3)
                rewards += self.eta * intrinsic_reward # intrinsic reward is just derived from forwards loss
                total_rewards.append(rewards.mean())  # Track mean total reward per step

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # Add to rollout buffer with next_obs for ICM
            if self.icm is not None:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    next_obs=new_obs,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        # Log ICM metrics averaged over the rollout
        if self.icm is not None:
            assert icm_forward_losses
            assert total_rewards
            self.last_mean_forward_loss = np.mean(icm_forward_losses)

            self.logger.record("train/icm_forward_loss", np.mean(icm_forward_losses))
            self.logger.record("train/icm_inverse_loss", np.mean(icm_inverse_losses))
            self.logger.record("train/mean_total_reward", np.mean(total_rewards))

        callback.on_rollout_end()

        return True


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        if self.icm is not None:
            # Update joint optimizer learning rate
            update_learning_rate(self.joint_optimizer, self.lr_schedule(self._current_progress_remaining))
            self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
        else:
            # Update policy optimizer learning rate (standard PPO)
            self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        if isinstance(self.clip_range, float):
            clip_range = self.clip_range
        else:
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # ICM update step using stored next_observations
                if self.icm is not None:
                    s1 = rollout_data.observations
                    s2 = rollout_data.next_observations  # Use stored next observations
                    phi_pred, actions_pred, forwards_loss, inverse_loss = self.icm(s1, s2, actions)

                    # Log phi_pred tensor values
                    # with open('phi_pred_tensors.log', 'a') as f:
                    #     f.write("\n\n----------------------------")
                    #     f.write(f"{phi_pred.detach().cpu().numpy()}\n")

                    # icm_loss = (1 - self.beta) * inverse_loss + self.beta * forwards_loss
                    # self.icm_optim.zero_grad()
                    # icm_loss.backward()
                    # self.icm_optim.step()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
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
                if self.icm is not None:
                    # perform joint update
                    loss = self._lambda * loss + (1 - self.beta) * inverse_loss + self.beta * forwards_loss

                    self.joint_optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients for both policy and ICM parameters
                    th.nn.utils.clip_grad_norm_(
                        chain(self.policy.parameters(), self.icm.parameters()),
                        self.max_grad_norm
                    )
                    self.joint_optimizer.step()
                else:
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
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