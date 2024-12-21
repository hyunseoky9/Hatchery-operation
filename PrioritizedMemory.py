import numpy as np
import random

# SumTree class
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
        if parent != 0:]
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):  # Leaf node
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total_sum(self):
        return self.tree[0]  # Root node contains the total sum