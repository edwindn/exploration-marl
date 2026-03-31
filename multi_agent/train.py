import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from encoder import utils
from encoder.world_model import WorldModel
from encoder.layers import api_model_conversion
from tensordict import TensorDict

# Import Dreamer utilities for actor-critic training
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from encoder.utils import computeLambdaValues, Moments

from encoder.buffer import Buffer
from local_nets import Actor, Critic
from maenv import GridEnv


class Logger:
	"""Placeholder logger class - does not log anything for now."""
	def log(self, metrics, mode):
		pass

	def finish(self, agent):
		pass


class Trainer(torch.nn.Module):
	"""
	World Model with Actor-Critic training.
	Uses implicit world model from TD-MPC2 for representation learning,
	combined with actor-critic policy learning similar to Dreamer.
	"""

	def __init__(self, env, cfg):
		super().__init__()
		self.env = env
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)

		# Actor and Critic networks
		self.actor = Actor(cfg, discrete_actions=cfg.discrete_actions, device=self.device).to(self.device)
		self.critic = Critic(cfg).to(self.device)

		# World model optimizer (encoder, dynamics, reward, termination)
		self.world_model_optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr, capturable=True)

		# Separate optimizers for actor and critic
		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr, capturable=True)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr, capturable=True)

		# Value normalization for advantage computation
		self.value_moments = Moments(self.device)

		self.model.eval()
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)

		# Training infrastructure
		self._step = 0
		self._ep_idx = 0
		self._tds = []
		self.buffer = Buffer(cfg)
		self.logger = Logger()

		if cfg.compile:
			print('Compiling training functions with torch.compile...')
			self.train_world_model = torch.compile(self.train_world_model, mode="reduce-overhead")
			self.train_policy = torch.compile(self.train_policy, mode="reduce-overhead")

	def _one_hot_encode_action(self, action):
		"""
		One-hot encode action if discrete, otherwise return as-is.

		Args:
			action (torch.Tensor): Action tensor (can be int indices or continuous).

		Returns:
			torch.Tensor: One-hot encoded action if discrete, otherwise original action.
		"""
		if self.cfg.discrete_actions:
			# Action should be integer indices, convert to one-hot
			if action.ndim == 1:
				action = action.unsqueeze(-1)
			action_min = action.min()
			action_max = action.max()
			assert action_min >= 0 and action_max < self.cfg.num_actions, (
				f"Discrete actions must be in [0, {self.cfg.num_actions-1}], "
				f"got min={action_min.item()} max={action_max.item()}"
			)
			# note actions are 1-indexed
			action_one_hot = F.one_hot(action.long(), num_classes=self.cfg.num_actions).float().squeeze(-2)
			return action_one_hot
		return action

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		elif isinstance(obs, np.ndarray):
			obs = torch.from_numpy(obs).unsqueeze(0)
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		elif isinstance(action, np.ndarray):
			action = torch.from_numpy(action)
		else:
			action = torch.tensor([action])
		if reward is None:
			reward = torch.tensor(float('nan'))
		elif isinstance(reward, float):
			reward = torch.tensor([reward], dtype=torch.float32)
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		elif not isinstance(terminated, torch.Tensor):
			terminated = torch.tensor([terminated], dtype=torch.float32)
		if action.ndim == 0:
			action = action.unsqueeze(0)
		td = TensorDict(
			obs=obs,
			action=action,
			reward=reward,
			terminated=terminated,
		batch_size=(1,))
		return td

	def common_metrics(self):
		"""Return common training metrics (step counter, buffer info, etc.)."""
		return {
			'step': self._step,
			'episode': self._ep_idx,
			'buffer_size': self.buffer.num_eps if hasattr(self.buffer, 'num_eps') else 0,
		}

	def _metric_to_scalar(self, value):
		"""Convert tensors and numpy scalars to plain Python scalars when possible."""
		if isinstance(value, torch.Tensor):
			if value.numel() == 1:
				return value.detach().cpu().item()
			return None
		if isinstance(value, np.ndarray):
			if value.size == 1:
				return value.item()
			return None
		if isinstance(value, (int, float)):
			return value
		return None

	def _log_metrics(self, train_metrics):
		"""Print a concise summary of the main training metrics."""
		if not train_metrics:
			return

		priority_keys = [
			"step",
			"episode",
			"buffer_size",
			"episode_reward",
			"episode_length",
			"wm_total_loss",
			"wm_consistency_loss",
			"wm_reward_loss",
			"wm_termination_loss",
			"actor_loss",
			"critic_loss",
			"entropy",
			"logprobs",
			"advantages",
			"critic_values",
			"wm_grad_norm",
			"actor_grad_norm",
			"critic_grad_norm",
		]

		parts = []
		for key in priority_keys:
			if key not in train_metrics:
				continue
			scalar = self._metric_to_scalar(train_metrics[key])
			if scalar is None:
				continue
			if isinstance(scalar, int):
				parts.append(f"{key}={scalar}")
			else:
				parts.append(f"{key}={scalar:.4f}")

		if not parts:
			return

		print("train | " + " | ".join(parts) + "\n")

	@torch.no_grad()
	def act(self, obs, eval_mode=False, task=None):
		"""
		Select an action using the actor network.

		Args:
			obs (torch.Tensor): Observation from the environment.
			eval_mode (bool): Whether to use deterministic action (mean).
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Encode observation to latent state
		if isinstance(obs, np.ndarray):
			obs = torch.from_numpy(obs)

		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)

		z = self.model.encode(obs, task)

		# Get action from actor
		action_dist = self.actor(z, task)

		if eval_mode:
			# Deterministic action (mean)
			if isinstance(action_dist, Normal):
				action = action_dist.mean
			else:  # Categorical
				action = torch.argmax(action_dist.probs, dim=-1)
		else:
			# Stochastic action
			action = action_dist.sample()

		return action[0].cpu()

	def train_policy(self, zs, actions, rewards, task=None):
		"""
		Actor-Critic policy update on encoded latent states from real buffer data.
		Based on Dreamer's behavior training with lambda returns.

		Args:
			zs (torch.Tensor): Sequence of latent states (horizon+1, batch_size, latent_dim).
			actions (torch.Tensor): Actions taken (horizon, batch_size, action_dim).
			rewards (torch.Tensor): Rewards received (horizon, batch_size, 1).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			TensorDict: Actor-critic training metrics.
		"""

		# Flatten time-major sequences for network evaluation, then restore to (batch, time).
		horizon_plus_1, batch_size, latent_dim = zs.shape
		zs_flat = zs.view(-1, latent_dim)
		horizon = horizon_plus_1 - 1

		# Get actions and log probabilities from actor on all states
		# Note: We compute log probs for the actions that were actually taken
		action_dist = self.actor(zs_flat[:-batch_size], task)  # Skip last timestep for action logprobs

		# Score the stored actions under the current policy without resampling.
		if self.cfg.discrete_actions:
			actions_flat = actions.reshape(horizon, batch_size, -1)
			if actions_flat.shape[-1] != 1:
				raise ValueError(
					f"Expected singleton discrete action dimension, got {actions_flat.shape}"
				)
			actions_flat = actions_flat.squeeze(-1).reshape(-1).long()
			logprobs = action_dist.log_prob(actions_flat).view(horizon, batch_size).transpose(0, 1)
			entropies = action_dist.entropy().view(horizon, batch_size).transpose(0, 1)
		else:
			actions_flat = actions.reshape(horizon, batch_size, -1)
			if actions_flat.shape[-1] != self.cfg.action_dim:
				raise ValueError(
					f"Expected continuous action dim {self.cfg.action_dim}, got {actions_flat.shape}"
				)
			actions_flat = actions_flat.reshape(-1, self.cfg.action_dim)
			logprobs = action_dist.log_prob(actions_flat).sum(-1).view(horizon, batch_size).transpose(0, 1)
			entropies = action_dist.entropy().sum(-1).view(horizon, batch_size).transpose(0, 1)

		# Get value predictions for all states
		values = self.critic(zs_flat, task).view(horizon_plus_1, batch_size).transpose(0, 1)  # (batch, horizon+1)

		# Get predicted rewards from world model (already computed during world model training)
		# For now, use actual rewards from buffer
		predicted_rewards = rewards.squeeze(-1).transpose(0, 1)  # (batch, horizon)

		# Compute continues (discount factors)
		if self.cfg.episodic:
			# Use termination predictor to get continuation probability
			continues = 1.0 - self.model.termination(zs_flat, task).view(batch_size, horizon_plus_1)
			continues = continues[:, :-1] * self.discount  # (batch, horizon)
		else:
			continues = torch.full_like(predicted_rewards, self.discount)

		# Compute lambda returns (GAE-style)
		lambda_values = computeLambdaValues(
			predicted_rewards,
			values,
			continues,
			self.cfg.lambda_
		)  # (batch, horizon)

		# Normalize advantages
		_, inverse_scale = self.value_moments(lambda_values)
		advantages = (lambda_values - values[:, :-1]) / inverse_scale

		# Actor loss: policy gradient with entropy bonus
		actor_loss = -torch.mean(
			advantages.detach() * logprobs +
			self.cfg.entropy_coef * entropies
		)

		self.actor_optim.zero_grad()
		actor_loss.backward()
		actor_grad_norm = torch.nn.utils.clip_grad_norm_(
			self.actor.parameters(),
			self.cfg.grad_clip_norm
		)
		self.actor_optim.step()

		# Critic loss: predict lambda returns
		value_preds = self.critic(zs_flat[:-batch_size].detach(), task).view(horizon, batch_size).transpose(0, 1)
		critic_loss = F.mse_loss(value_preds, lambda_values.detach())

		self.critic_optim.zero_grad()
		critic_loss.backward()
		critic_grad_norm = torch.nn.utils.clip_grad_norm_(
			self.critic.parameters(),
			self.cfg.grad_clip_norm
		)
		self.critic_optim.step()

		# Return metrics
		return TensorDict({
			"actor_loss": actor_loss,
			"critic_loss": critic_loss,
			"actor_grad_norm": actor_grad_norm,
			"critic_grad_norm": critic_grad_norm,
			"entropy": entropies.mean(),
			"logprobs": logprobs.mean(),
			"advantages": advantages.mean(),
			"critic_values": values.mean(),
		})

	def train_world_model(self, obs, action, reward, terminated, task=None):
		"""
		Train the world model (encoder, dynamics, reward, termination).
		Uses consistency loss to train implicit latent dynamics.

		Args:
			obs (torch.Tensor): Observations (horizon+1, batch_size, obs_dim).
			action (torch.Tensor): Actions (horizon, batch_size, action_dim).
			reward (torch.Tensor): Rewards (horizon, batch_size, 1).
			terminated (torch.Tensor): Termination flags (horizon, batch_size, 1).
			task (torch.Tensor): Task indices (only for multi-task).

		Returns:
			tuple: (latent_states, metrics_dict)
		"""
		# Encode next observations for consistency targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)

		# Prepare for update
		self.model.train()

		# Latent rollout: predict future latents using dynamics model
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			_action = _action.unsqueeze(-1)
			# One-hot encode action if discrete before passing to world model
			_action_encoded = self._one_hot_encode_action(_action)
			z = self.model.next(z, _action_encoded, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Reward prediction
		_zs = zs[:-1]
		# One-hot encode actions if discrete before passing to world model
		action = action.unsqueeze(-1)
		action_encoded = self._one_hot_encode_action(action)
		reward_preds = self.model.reward(_zs, action_encoded, task)
		reward_loss = 0
		for t, (rew_pred, rew) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0))):
			reward_loss = reward_loss + utils.soft_ce(rew_pred, rew, self.cfg).mean() * self.cfg.rho**t

		# Termination prediction (if episodic)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = torch.tensor(0.0, device=self.device)

		# Normalize losses
		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon

		# Total world model loss
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss
		)

		# Update world model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.world_model_optim.step()
		self.world_model_optim.zero_grad(set_to_none=True)

		# Return training statistics and latent states
		self.model.eval()
		metrics = TensorDict({
			"wm_consistency_loss": consistency_loss,
			"wm_reward_loss": reward_loss,
			"wm_termination_loss": termination_loss,
			"wm_total_loss": total_loss,
			"wm_grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			metrics.update(utils.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))

		return zs.detach(), action, reward, metrics

	def update(self, buffer):
		"""
		Main update function. Trains both world model and actor-critic policy.

		Args:
			buffer: Replay buffer with sample() method.

		Returns:
			TensorDict: Combined training statistics from both updates.
		"""
		# Sample batch from buffer
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task

		torch.compiler.cudagraph_mark_step_begin()

		# Train world model and get latent states
		zs, actions_used, rewards_used, world_model_metrics = self.train_world_model(
			obs, action, reward, terminated, **kwargs
		)

		# Train actor-critic policy on latent states from real data
		policy_metrics = self.train_policy(zs, actions_used, rewards_used, **kwargs)

		# Combine metrics
		all_metrics = TensorDict({
			**world_model_metrics,
			**policy_metrics
		})

		return all_metrics.mean()

	def train(self):
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:

			if done or (self._step + 1) % cfg.collect_every == 0:

				if self._step > 0:
					# if info['terminated'] and not self.cfg.episodic:
					# 	raise ValueError('Termination detected but you are not in episodic mode. ' \
					# 	'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						# episode_success=info['success'],
						episode_length=len(self._tds),
						# episode_terminated=info['terminated'],
						)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs, _ = self.env.reset()
				self._tds = []
				#self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step >= self.cfg.seed_steps:
				action = self.act(obs)
			else:
				action = self.env.rand_act()
			obs, reward, term, trunc, info = self.env.step(action.item())
			done = term or trunc
			self._tds.append(self.to_td(obs, action, reward, term))

			# Update agent (world model + actor-critic)
			if self._step > self.cfg.seed_steps:
				_train_metrics = self.update(self.buffer)
				train_metrics.update(_train_metrics)

			if (self._step + 1) % cfg.print_every == 0:
				self._log_metrics(train_metrics)

			self._step += 1

		self.logger.finish(self)
	
	def close(self):
		pass


if __name__ == "__main__":
	import yaml
	from types import SimpleNamespace

	# Load configuration
	config_path = os.path.join(os.path.dirname(__file__), 'train_config.yaml')
	with open(config_path, 'r') as f:
		config_dict = yaml.safe_load(f)

	env = GridEnv()

	config_dict["obs_shape"] = {config_dict["obs"]: env.observation_space.shape}
	config_dict["num_channels"] = env.observation_space.shape[-1]

	cfg = SimpleNamespace(**config_dict)

	trainer = Trainer(env, cfg)
	trainer.train()
	trainer.close()
