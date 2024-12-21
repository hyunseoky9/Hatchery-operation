import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Determines how much prioritization is applied
        self.epsilon = 1e-6  # Small value to avoid zero priority

    def add(self, error, transition):
        # Priority is proportional to TD error
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=0.4):
        mini_batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_sum() / batch_size  # Divide the total sum into segments
        for i in range(batch_size):
            r = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.sample(r)
            mini_batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        sampling_probs = priorities / self.tree.total_sum()
        weights = (1 / (len(self.tree.data) * sampling_probs)) ** beta
        weights /= weights.max()  # Normalize weights
        return mini_batch, idxs, weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

# SumTree class for storing priority 
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # Sum-tree array
        self.data = np.zeros(capacity, dtype=object)  # Store transitions
        self.write = 0  # Pointer to overwrite old data

    def add(self, priority, data):
        idx = self.write + self.capacity - 1  # Index in the leaf node
        self.data[self.write] = data  # Store data
        self.update(idx, priority)  # Update the tree with the new priority

        self.write += 1
        if self.write >= self.capacity:  # Overwrite old data
            self.write = 0

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def total_sum(self):
        return self.tree[0]  # Root node contains the total sum