import numpy as np
import random 
class Nstepqueue:
    """
    Implementation of n-step queue for n-step DQN.
    This class maintains a queue of transitions and calculates the n-step return for each transition.
    """
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = []  # Temporary queue to hold transitions for n-step calculation
        self.rqueue = []  # Temporary queue to hold rewards for calculating cumulative rewards

    def add(self, state, action, reward, next_state, done, previous_action, memory, per):
        # Add to n-step queue
        # If the queue becomes length of n, add a transition to memory.
        # Also, if the episode is done, add rest of the queue to memory.
        transition = (state, action, reward, next_state, done, previous_action)
        self.queue.append(transition)
        self.rqueue.append(reward)
        # If n-step queue is ready, calculate n-step return
        if len(self.queue) >= self.n:
            self.add2mem(memory, per)
        
        # If the episode is done, clear the n-step queue
        if done:
            while len(self.queue) > 0:
                self.add2mem(memory, per)
            self.queue = [] # make sure the queue is cleared after the episode is done
            self.rqueue = []
            
    def add2mem(self, memory, per):
        G = sum(self.gamma**np.arange(len(self.queue)) * self.rqueue)
        state, action, _, _, _, previous_action = self.queue[0]  # Take the first state-action pair
        _, _, _, next_state, done, _ = self.queue[-1]  # Take the last next_state and done
        if per: # prioritized experience replay
            memory.add(memory.max_abstd, (state, action, G, next_state, done, previous_action)) # add experience to memory
        else: # vanilla experience replay
            memory.add(state, action, G, next_state, done, previous_action) # add experience to memory
        self.queue.pop(0) # Remove the oldest transition from the n-step queue
        self.rqueue.pop(0) # Remove the oldest transition from the n-step queue