import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor


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

    def sample(self, value):
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):  # Leaf node
            return idx
        if value < self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total_sum(self):
        return self.tree[0]  # Root node contains the total sum
    

class PMemory:
    def __init__(self, capacity, alpha, epsilon, max_abstd):
        self.buffer_size = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Determines how much prioritization is applied
        self.epsilon = epsilon  # Small value to avoid zero priority
        self.max_abstd = max_abstd # maximum absolute td
        
    def add(self, error, transition):
        # Priority is proportional to TD error
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta):
        mini_batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_sum() / batch_size  # Divide the total sum into segments
        for i in range(batch_size):
            #r = rng[i]*self.tree.total_sum() # for testing parallel with vanilla
            r = np.random.uniform(segment * i, segment * (i + 1))
            #r = np.random.uniform(0, self.tree.total_sum())
            idx, priority, data = self.tree.sample(r)
            mini_batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        sampling_probs = priorities / self.tree.total_sum()
        weights = (1 / (len(self.tree.data) * sampling_probs)) ** beta
        weights /= weights.max()  # Normalize weights
        return mini_batch, idxs, weights

    def _sample_one(self, r):
        return self.tree.sample(r)

    def sample_parallel(self, batch_size, beta):
        segment = self.tree.total_sum() / batch_size
        segment_values = [np.random.uniform(i * segment, (i + 1) * segment) for i in range(batch_size)]

        # Use ThreadPoolExecutor to parallelize sampling
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._sample_one, segment_values))
        idxs, priorities, mini_batch = zip(*results)
        # calculate weights for importance sampling bias correction
        sampling_probs = np.array(priorities) / self.tree.total_sum()
        weights = (1 / (len(self.tree.data) * sampling_probs)) ** beta
        weights /= weights.max()

        return list(mini_batch), list(idxs), weights
    
    def update_priorities(self, idxs, abserrors):
        for idx, abserrors in zip(idxs, abserrors):
            priority = (abserrors + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
