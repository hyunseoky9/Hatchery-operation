import numpy as np
from math import floor
import random
from IPython.display import display
import pandas as pd
class Env0_0:
    def __init__(self,initstate,parameterization_set,discretization_set):
        self.envID = 'Env0.0'
        # Define state space and action space based on your document
        if discretization_set == 0:
            self.states = {
                "NW": [1000, 111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 1000000], # Population size
                "q": [0,1,2,3,4], # Spring Flow (index, not actual values of cfs)
            }
            self.actions = {
                "a": [0, 100000, 200000, 300000]
                #"a": [0, 3333, 66666, 100000, 133333, 166666, 200000, 233333, 266666, 300000]
            }
        elif discretization_set == 1:
            self.states = {
                "NW": [1000, 3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "q": [0,1,2,3,4], # Spring Flow (index, not actual values of cfs)
            }
            self.actions = {
                "a": [0, 75000, 150000, 225000, 300000]
            }
        
        self.statespace_dim = list(map(lambda x: len(x[1]), self.states.items()))
        self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))

        # IMPORTANT INFO:
        # statespace: 7*6*5*7*5*2 = 14700
        # actionspace: 7*6*5*7*5*2*5 = 73500
        # reachable state space reduced b/c only 1 spring flow state possible when tau=1
        # & only 1 hatchery state possible when tau=0
        # reachable state space: 7*6*1*7*5 + 7*6*5*7*1 = 2940
        # reachable state-action space reduced b/c fall action cannot be greater than NH
        # reachable state-action space 7*7*1*7*5*5 + sum_{i=1}^{5} 7*7*1*7*1*i = 11760

        

        # Initialize state
        self.state = self.reset(initstate)
        # Define parameters
        # call in parameterization dataset csv
        # Read csv 'parameterization_env1.0.csv'
        self.parset = parameterization_set - 1
        parameterization_set_filename = 'parameterization_env0.0.csv'
        paramdf = pd.read_csv(parameterization_set_filename)
        self.b0 = paramdf['b0'][parameterization_set - 1]
        self.b1 = paramdf['b1'][parameterization_set - 1]
        self.muq = paramdf['muq'][parameterization_set - 1]        
        self.sigq = paramdf['sigq'][parameterization_set - 1]        
        self.d = paramdf['d'][parameterization_set - 1] # death rate
        self.d1 = paramdf['d1'][parameterization_set - 1] # death rate
        self.Nth = paramdf['Nth'][parameterization_set - 1]        
        self.p = paramdf['p'][parameterization_set - 1] # Reward for persistence
        self.c = paramdf['c'][parameterization_set - 1] # Cost for augmentation
        self.gamma = paramdf['gamma'][parameterization_set - 1] # Discount factor
        self.extpenalty = paramdf['extpenalty'][parameterization_set - 1] # Penalty for extinction


    def reset(self, initstate):
        # Initialize state variables
        new_state = []
        if initstate[0] == -1:
            new_state.append(random.choice(np.arange(1, len(self.states["NW"])))) # don't start from the smallest population size
        else:
            new_state.append(initstate[0])
        if initstate[1] == -1:
            new_state.append(random.choice(np.arange(0, len(self.states["q"]))))
        else: 
            new_state.append(initstate[1])
        self.state = new_state
        return self.state

    def step(self, action):
        # Compute next state and reward based on action and transition rules
        NW = self.states["NW"][self.state[0]]
        q = self.states["q"][self.state[1]]
        a = self.actions["a"][action]
        # Check termination
        if NW > self.Nth:  # Extinction threshold
            r = (self.b0 + self.b1*q) - (self.d*(1+self.d1*(min(a/NW,1))))
            NW_next = np.exp(r)*(NW+a)
            reward = self.p - self.c if a > 0 else self.p
            q_next =  max(np.random.normal(self.muq, self.sigq),0)

            # put it into defined discrete states
            NW_next = self._discretize(NW_next, self.states['NW'])
            q_next = self._discretize(q_next, self.states['q'])
            # Update state
            NW_next_idx = np.where(np.array(self.states['NW']) == NW_next)[0][0]
            q_next_idx = np.where(np.array(self.states['q']) == q_next)[0][0]
            self.state = [NW_next_idx, q_next_idx]
            # Check termination
            done  = False
        else:
            # extinction
            done = True
            reward = self.extpenalty
            r = 0

        return reward, done, r

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

    def _flatten(self,stateidx):
        # change factored state index in to a single index
        statespace_dim = list(map(lambda x: len(x[1]), self.states.items()))
        flattened_index = 0
        stride = 1
        for i, s in zip(reversed(stateidx), reversed(statespace_dim)):
            flattened_index += i * stride
            stride *= s
        return flattened_index

    def _unflatten(self,stateid):
        # change single index into factored state index
        indices = []
        for s in reversed(self.statespace_dim):
            indices.append(stateid % s)
            stateid //= s
        return list(reversed(indices))
