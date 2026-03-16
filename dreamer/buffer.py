import attridict
import numpy as np
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.nextObservations   = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, actions_size), dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones               = np.empty((self.capacity, 1), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward, nextObservation, done):
        self.observations[self.bufferIndex]     = observation
        self.actions[self.bufferIndex]          = action
        self.rewards[self.bufferIndex]          = reward
        self.nextObservations[self.bufferIndex] = nextObservation
        self.dones[self.bufferIndex]            = done

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"
        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations         = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        nextObservations    = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()

        actions  = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards  = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones    = torch.as_tensor(self.dones[sampleIndex], device=self.device)

        sample = attridict({
            "observations"      : observations,
            "actions"           : actions,
            "rewards"           : rewards,
            "nextObservations"  : nextObservations,
            "dones"             : dones})
        return sample