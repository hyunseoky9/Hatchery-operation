import numpy as np
from math import floor
import random
from IPython.display import display
import pandas as pd

class Tiger:
    """
    Tiger problem from Iadine's pnas paper. It describes a POMDP about conserving sumatran tiger.
    """
    def __init__(self,initstate):
        self.envID = 'tiger'
        self.partial = True
        self.episodic = False
        self.absorbing_cut = True # has an absorbing state and the episode should be cut shortly after reaching it.
        self.discset = 0
        self.parset = 0

        # Define state space and action space based on your document
        self.states = {
            "existing": [0, 1]  # 0= extant, 1= extinct
        }
        self.observations = {
            "O": [0, 1]  # 0= absent, 1=present
        }
        self.actions = {
            "a": [0,1,2] # 0= do nothing, 1= manage, 2= survey
        }

        self.statespace_dim = list(map(lambda x: len(x[1]), self.states.items()))
        self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
        self.obsspace_dim  = list(map(lambda x: len(x[1]), self.observations.items()))


        # parameters
        self.gamma = 0.95 # Discount factor
        self.default_obs_prob = [0.99,0.01]
        self.survey_obs_prob =  [0.0,1.0] #[0.78193,0.21807]
        # Initialize state and observation
        self.state, self.obs = self.reset(initstate)
        print(f'observation probability: {self.default_obs_prob}, survey observation probability: {self.survey_obs_prob}')

    def reset(self,initstate):
        new_state = []
        new_obs = []
        if initstate[0] == -1:
            new_state.append(1)
            new_obs.append(1)
        else:
            new_state.append(initstate[0])

        self.state = new_state
        self.obs = new_obs
        return self.state, self.obs

    def step(self, action):
        extant = self.state[0]
        
        if action == 0: # do nothing
            if extant == 0:
                reward = 0
                next_state = [0]
                done = False
            else:
                reward = 175.133
                next_state = [int(np.random.choice([0,1], p=[0.1,0.9]))]
                done = False
            if next_state[0] == 1:
                next_obs = [int(np.random.choice([0,1], p=self.default_obs_prob))]
            else:
                next_obs = [0]
        elif action == 1: # manage
            if extant == 0:
                reward = -18.784
                next_state = [0]
                done = False
            else:
                reward = 156.349
                next_state = [int(np.random.choice([0,1], p=[0.05816,0.94184]))]
                done = False
            if next_state[0] == 1:
                next_obs = [int(np.random.choice([0,1], p=self.default_obs_prob))]
            else:
                next_obs = [0]

        else: # survery
            if extant == 0:
                reward = -10.84
                next_state = [0]
                done = False 
            else:
                reward = 164.293
                next_state = [int(np.random.choice([0,1], p=[0.1,0.9]))]
                done = False
            if next_state[0] == 1:
                next_obs = [int(np.random.choice([0,1], p=self.survey_obs_prob))]
            else:
                next_obs = [0]

        self.state = next_state
        self.obs = next_obs

        return reward, done, 0

        




