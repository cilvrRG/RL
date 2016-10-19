from collections import deque
import numpy as np


class replay_buffer(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.deque = deque()

        self.obserbation_batch = np.array([])
        self.action_batch = np.array([])
        self.reward_batch = np.array([])
        self.next_obserbation_batch = np.array([])
        self.mask_batch = np.array([])

    def insert(self, obs, action, reward, newobs, mask):
        self.deque.append([obs, action, reward, newobs, mask])
        if len(self.deque) > self.max_size:
            self.deque.popleft()

    def sample(self, batch_size):
        indices = np.random.choice(len(self.deque), batch_size)

        self.obserbation_batch.resize(batch_size, self.deque[0][0].shape[0])
        self.action_batch.resize(batch_size, 1)
        self.reward_batch.resize(batch_size, 1)
        self.next_obserbation_batch.resize(
            batch_size, self.deque[0][0].shape[0])
        self.mask_batch.resize(batch_size, 1)

        for b in range(batch_size):
            other_b = indices[b]
            self.obserbation_batch[b] = self.deque[other_b][0]
            self.action_batch[b] = self.deque[other_b][1]
            self.reward_batch[b] = self.deque[other_b][2]
            self.next_obserbation_batch[b] = self.deque[other_b][3]
            self.mask_batch[b] = self.deque[other_b][4]

        return self.obserbation_batch, self.action_batch, self.reward_batch, self.next_obserbation_batch, self.mask_batch
