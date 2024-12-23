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

    def add(self, state, action, reward, next_state, done, memory, per):
        # Add to n-step queue
        self.queue.append((state, action, reward, next_state, done))
        print(f'queue({len(self.queue)}): {self.queue}')
        # If n-step queue is ready, calculate n-step return
        if len(self.queue) >= self.n:
            self.add2mem(memory, per)
            print('added 2 mem')
        
        # If the episode is done, clear the n-step queue
        if done:
            while len(self.queue) > 0:
                self.add2mem(memory, per)
                print('added 2 mem')
            self.queue = [] # make sure the queue is cleared after the episode is done
        print(f'queue after memory add({len(self.queue)}): {self.queue}')
            
    def add2mem(self, memory, per):
        G = sum([self.gamma**i * self.queue[i][2] for i in range(len(self.queue))])  # Accumulate rewards
        state, action, _, _, _ = self.queue[0]  # Take the first state-action pair
        _, _, _, next_state, done = self.queue[-1]  # Take the last next_state and done
        if per: # prioritized experience replay
            memory.add(memory.max_abstd, (state, action, G, next_state, done)) # add experience to memory
        else: # vanilla experience replay
            memory.add(state, action, G, next_state, done) # add experience to memory
        self.queue.pop(0) # Remove the oldest transition from the n-step queue