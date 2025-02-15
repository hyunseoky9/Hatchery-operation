import numpy as np
from math import floor
import random
from IPython.display import display
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class MyDummyEnv(gym.Env):
    """
    A Gymnasium-compatible version of Env2_1.
    """
    def __init__(self,config=None):
        super(MyDummyEnv, self).__init__()
        # Load configuration
        if config is None:
            config = {}
        # Safe defaults
        initstate = config.get("initstate", [-1, -1, -1, -1, -1, -1])
        parameterization_set = config.get("parameterization_set", 2)
        discretization_set = config.get("discretization_set", 0)

        self.envID = 'Env2.1'
        self.partial = True
        self.episodic = False
        self.absorbing_cut = True # has an absorbing state and the episode should be cut shortly after reaching it.
        self.discset = discretization_set
        # Define state space and action space based on your document
        if discretization_set == 0:
            self.states = {
                "NW": [1000, 3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "NWm1": [3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "NH": [0, 75000, 150000, 225000, 300000], # hatchery population size
                "H": [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86], # Heterozygosity
                "q": [65, 322, 457, 592, 848], # Spring Flow
                "tau": [0, 1]  # 0 for Fall, 1 for Spring
            }
            self.observations = {
                "ONW": [1000, 3000, 15195, 76961, 389806, 1974350, 10000000], # population size
                "ONWm1": [3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "ONH": [0, 75000, 150000, 225000, 300000], # observed hatchery fish
                "OH": [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86], # observed heterozygosity
                "Oq": [65, 322, 457, 592, 848], # observed spring flow
                "Otau": [0, 1]  # observed season
            }
            self.actions = {
                "a": [0, 75000, 150000, 225000, 300000]
            }


        # observed catch from fall monitoring. -1= no observed catch (for spring); 45 is actually anything gretaer than 45
        self.statespace_dim = list(map(lambda x: len(x[1]), self.states.items()))
        self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
        self.obsspace_dim  = list(map(lambda x: len(x[1]), self.observations.items()))

        # Define Gymnasium spaces
        self.observation_space = spaces.Discrete(len(self.actions["a"]))
        self.action_space = spaces.Discrete(len(self.actions["a"]))

        # Specify if you want to support or define a maximum number of steps.
        self.max_steps = 100
        self.current_step = 0
        self.absorbing_counter = 0




    def reset(self, initstate=None, seed=None, options=None):
        return np.array([1]), {}
    
    def step(self, action):
        # Compute next state and reward based on action and transition rules
        return np.array([1]), 1.0, False, False, {}
        

    
    def _fallmonitoring(self, NW):
        # fall monitoring
        expcatch = self.theta * NW
        return np.random.negative_binomial(self.sigy,  self.sigy/(expcatch + self.sigy))

    def _survival_rate(self, H):
        # compute survival rate based on heterozygosity
        eps = np.random.normal(0, self.sigs)
        return np.exp(-max(self.alpha0 + self.alpha1 * (1 - H) + eps, 0))

    def _recruitment_rate(self, q):
        # compute recruitment rate based on spring flow
        return self.beta*q

    def _update_heterozygosity(self, H, NW, a1):
        # Update heterozygosity based on the number of hatchery fish that's stocked
        return (a1 * H * (1 - self.l) + NW * H) / (a1 + NW) # weighted average of hatchery and wild heterozygosity

    def _nextgen_heterozygosity(self, H,NW,NWm1):
        # Compute heterozygosity of the next generation
        Ne = 2/(1/NW + 1/NWm1)
        mode = H * (1 + (self.mu - 1 / (2 * Ne)))
        a = mode*(self.kappa - 2) + 1
        b = (1 - mode)*(self.kappa - 2) + 1
        return np.random.beta(a, b)
    
    def _discretize(self, x, possible_states):
        # get the 2 closest values in the possible_states tox 
        # and then get the weights disproportionate to the distance to the x
        # then use those weights as probabilities to chose one of the two states.
        if x < possible_states[0]:
            return possible_states[0]
        elif x > possible_states[-1]:
            return possible_states[-1]
        else:
            lower, upper = np.array(possible_states)[np.argsort(abs(np.array(possible_states) - x))[:2]]
            weights = np.array([upper - x, x - lower])/(upper - lower)  
            return random.choices([lower, upper], weights=weights,k=1)[0]
