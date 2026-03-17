from __future__ import print_function
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LSTMPolicy
from icm import StateActionPredictor, StatePredictor
import scipy.signal
from constants import constants

def discount(x, gamma):
    """
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0, clip=False):
    """
    Given a rollout, compute its returns and the advantage.
    """
    # collecting transitions
    if rollout.unsup:
        batch_si = np.asarray(rollout.states + [rollout.end_state])
    else:
        batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(rollout.bonuses + [0])
    if clip:
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # value network target

    # collecting target for policy network
    rewards = np.asarray(rollout.rewards)
    if rollout.unsup:
        rewards += np.asarray(rollout.bonuses)
    if clip:
        rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, unsup=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.unsup = unsup
        if self.unsup:
            self.bonuses = []
            self.end_state = None


    def add(self, state, action, reward, value, terminal, features,
                bonus=None, end_state=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        if self.unsup:
            self.bonuses += [bonus]
            self.end_state = end_state

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if self.unsup:
            self.bonuses.extend(other.bonuses)
            self.end_state = other.end_state

def collect_rollout(env, policy, num_local_steps, render, predictor,
                    envWrap, noReward, state, features):
    """
    Collect a single rollout from the environment.
    Returns the rollout and the updated state and features.
    """
    length = 0
    rewards = 0
    values = 0
    if predictor is not None:
        life_bonus = 0

    terminal_end = False
    rollout = PartialRollout(predictor is not None)

    for _ in range(num_local_steps):
        # run policy
        fetched = policy.act(state, *features)
        action, value_, new_features = fetched[0], fetched[1], fetched[2:]

        # run environment: get action_index from sampled one-hot 'action'
        stepAct = action.argmax()
        next_state, reward, terminal, info = env.step(stepAct)
        if noReward:
            reward = 0.
        if render:
            env.render()

        curr_tuple = [state, action, reward, value_, terminal, features]
        if predictor is not None:
            bonus = predictor.pred_bonus(state, next_state, action)
            curr_tuple += [bonus, next_state]
            life_bonus += bonus

        # collect the experience
        rollout.add(*curr_tuple)
        rewards += reward
        length += 1
        values += value_[0]

        state = next_state
        features = new_features

        timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        if timestep_limit is None: timestep_limit = env.spec.timestep_limit
        if terminal or length >= timestep_limit:
            # prints summary of each life if envWrap==True else each game
            if predictor is not None:
                print("Episode finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (rewards, length, life_bonus))
            else:
                print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
            if 'distance' in info: print('Mario Distance Covered:', info['distance'])
            terminal_end = True
            features = policy.get_initial_features()  # reset lstm memory
            # reset only if it hasn't already reseted
            if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                state = env.reset()
            break

    if not terminal_end:
        rollout.r = policy.value(state, *features)

    return rollout, state, features


class A2C(object):
    def __init__(self, env, task, visualise, unsupType, envWrap=False, designHead='universe', noReward=False):
        """
        An implementation of the A2C algorithm for single agent training.
        """
        self.task = task
        self.unsup = unsupType is not None
        self.envWrap = envWrap
        self.env = env
        self.visualise = visualise
        self.noReward = noReward
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        predictor = None
        numaction = env.action_space.n

        # Initialize networks
        self.network = LSTMPolicy(env.observation_space.shape, numaction, designHead).to(self.device)
        self.global_step = 0

        if self.unsup:
            if 'state' in unsupType:
                self.ap_network = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType).to(self.device)
            else:
                self.ap_network = StateActionPredictor(env.observation_space.shape, numaction, designHead).to(self.device)
            predictor = self.ap_network

        # Setup optimizer
        print("Optimizer: ADAM with lr: %f" % (constants['LEARNING_RATE']))
        print("Input observation shape: ", env.observation_space.shape)

        params = list(self.network.parameters())
        if self.unsup:
            params += list(self.ap_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=constants['LEARNING_RATE'])

        self.predictor = predictor
        self.num_local_steps = constants['ROLLOUT_MAXLEN']

        # Initialize environment state
        self.state = env.reset()
        self.features = self.network.get_initial_features()

        # Initialize extras
        self.local_steps = 0

    def compute_loss(self, batch):
        """
        Compute A3C loss given a batch of experiences.
        """
        # Prepare batch data
        if self.unsup:
            states = torch.FloatTensor(batch.si[:-1]).to(self.device)
        else:
            states = torch.FloatTensor(batch.si).to(self.device)

        actions = torch.FloatTensor(batch.a).to(self.device)
        advantages = torch.FloatTensor(batch.adv).to(self.device)
        returns = torch.FloatTensor(batch.r).to(self.device)
        features = (torch.FloatTensor(batch.features[0]).to(self.device),
                   torch.FloatTensor(batch.features[1]).to(self.device))

        # Forward pass through policy network
        logits, values = self.network(states, features)

        # Computing a3c loss: https://arxiv.org/abs/1506.02438
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # 1) the "policy gradients" loss: its derivative is precisely the policy gradient
        # adv will contain the advantages, as calculated in process_rollout
        pi_loss = -(log_probs * actions).sum(dim=-1) * advantages  # Eq (19)
        pi_loss = pi_loss.mean()

        # 2) loss of value function: l2_loss = (x-y)^2/2
        vf_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()  # Eq (28)

        # 3) entropy to ensure randomness
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # final a3c loss: lr of critic is half of actor
        loss = pi_loss + 0.5 * vf_loss - entropy * constants['ENTROPY_BETA']

        # computing predictor loss
        if self.unsup:
            s1 = torch.FloatTensor(batch.si[:-1]).to(self.device)
            s2 = torch.FloatTensor(batch.si[1:]).to(self.device)
            asample = torch.FloatTensor(batch.a).to(self.device)

            pred_loss = self.ap_network.compute_loss(s1, s2, asample)
            if hasattr(self.ap_network, 'invloss'):
                predloss = constants['PREDICTION_LR_SCALE'] * (
                    self.ap_network.invloss * (1 - constants['FORWARD_LOSS_WT']) +
                    self.ap_network.forwardloss * constants['FORWARD_LOSS_WT']
                )
            else:
                predloss = constants['PREDICTION_LR_SCALE'] * self.ap_network.forwardloss

            loss = loss + predloss

        # Scale by batch size (factored out to make hyperparams not depend on it)
        loss = loss * 20.0

        return loss

    def process(self):
        """
        Collect a rollout from the environment and update the parameters.
        """
        # Collect rollout
        rollout, self.state, self.features = collect_rollout(
            self.env, self.network, self.num_local_steps,
            self.visualise, self.predictor, self.envWrap,
            self.noReward, self.state, self.features
        )

        batch = process_rollout(rollout, gamma=constants['GAMMA'], lambda_=constants['LAMBDA'], clip=self.envWrap)

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute loss
        loss = self.compute_loss(batch)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), constants['GRAD_NORM_CLIP'])
        if self.unsup:
            torch.nn.utils.clip_grad_norm_(self.ap_network.parameters(), constants['GRAD_NORM_CLIP'])

        # Update parameters
        self.optimizer.step()

        # Update global step
        self.global_step += len(batch.si)

        if batch.terminal:
            print("Global Step Counter: %d" % self.global_step)

        self.local_steps += 1


if __name__ == "__main__":
    import yaml
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from environment.env import NavEnv

    # Load training configuration
    config_path = Path(__file__).parent / "train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract A2C hyperparameters
    a2c_config = config['a2c']
    training_config = config['training']

    # Update constants with config values
    constants['LEARNING_RATE'] = a2c_config['learning_rate']
    constants['GAMMA'] = a2c_config['gamma']
    constants['LAMBDA'] = a2c_config['gae_lambda']
    constants['ENTROPY_BETA'] = a2c_config['ent_coef']
    constants['ROLLOUT_MAXLEN'] = a2c_config['n_steps']

    # Set default values for constants not in config
    if 'GRAD_NORM_CLIP' not in constants:
        constants['GRAD_NORM_CLIP'] = 40.0
    if 'REWARD_CLIP' not in constants:
        constants['REWARD_CLIP'] = 1.0

    # Create environment
    env = NavEnv(render_mode=None)

    print("=" * 60)
    print("Starting A2C Training on NavEnv")
    print("=" * 60)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Learning rate: {constants['LEARNING_RATE']}")
    print(f"Gamma: {constants['GAMMA']}")
    print(f"GAE Lambda: {constants['LAMBDA']}")
    print(f"Entropy coefficient: {constants['ENTROPY_BETA']}")
    print(f"Rollout length: {constants['ROLLOUT_MAXLEN']}")
    print(f"Total timesteps: {training_config['total_timesteps']}")
    print("=" * 60)

    # Initialize A2C agent
    agent = A2C(
        env=env,
        task=0,
        visualise=False,
        unsupType=None,
        envWrap=False,
        designHead='universe',
        noReward=False
    )

    # Training loop
    try:
        total_steps = 0
        while total_steps < training_config['total_timesteps']:
            agent.process()
            total_steps = agent.global_step

            if agent.local_steps % 100 == 0:
                print(f"Training progress: {total_steps}/{training_config['total_timesteps']} steps")

        print("=" * 60)
        print("Training completed!")
        print(f"Final step count: {agent.global_step}")
        print("=" * 60)

        # Save the trained model
        save_path = Path(training_config['save_path'] + '_a2c.pth')
        torch.save({
            'network_state_dict': agent.network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'global_step': agent.global_step,
        }, save_path)
        print(f"Model saved to {save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Steps completed: {agent.global_step}")

    env.close()