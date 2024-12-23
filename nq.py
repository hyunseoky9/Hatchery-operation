import numpy as np
import random 
class nq:
    """
    Implementation of n-step queue for n-step DQN.
    This class maintains a queue of transitions and calculates the n-step return for each transition.
    """
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = []  # Temporary queue to hold transitions for n-step calculation

    def add(self, state, action, reward, next_state, done, memory, per):
        # Add to n-step queue
        self.queue.append((state, action, reward, next_state, done))
        
        # If n-step queue is ready, calculate n-step return
        if len(self.queue) >= self.n:
            G = sum([self.gamma**i * self.queue[i][2] for i in range(self.n)])  # Accumulate rewards
            state, action, _, _, _ = self.queue[0]  # Take the first state-action pair
            _, _, _, next_state, done = self.queue[-1]  # Take the last next_state and done
            if per: # prioritized experience replay
                memory.add(memory.max_abstd, (state, action, reward, next_state, done)) # add experience to memory
            else: # vanilla experience replay
                memory.add(state, action, reward, next_state, done) # add experience to memory
            
            # Remove the oldest transition from the n-step queue
            self.queue.pop(0)
        # Maintain buffer size
        if len(self.memory) > self.capacity:
            self.memory.pop(0)