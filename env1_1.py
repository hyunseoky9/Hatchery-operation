import numpy as np
from math import floor
import random
from IPython.display import display
import pandas as pd
class Env1_1:
    """
    derivative of Env1.0.
    State space is now continuous
    Action space is still discrete
    """
    def __init__(self,initstate,parameterization_set,discretization_set):
        self.envID = 'Env1.1'
        self.partial = False
        self.discset = discretization_set
        self.episodic = True
        self.contstate= False

        # Define parameters
        # call in parameterization dataset csv
        # Read csv 'parameterization_env1.0.csv'
        self.parset = parameterization_set - 1
        parameterization_set_filename = 'parameterization_env1.0.csv'
        paramdf = pd.read_csv(parameterization_set_filename)
        self.alpha0 = paramdf['alpha0'][parameterization_set - 1] # survival parameter 1
        self.alpha1 = paramdf['alpha1'][parameterization_set - 1] # survival parameter 2
        self.sigs = paramdf['sigs'][parameterization_set - 1] # standard deviation of survival noise
        self.mu = paramdf['mu'][parameterization_set - 1] # mutation rate
        self.kappa = paramdf['kappa'][parameterization_set - 1] # concentration for the beta distribution for heterozygosity
        self.beta = paramdf['beta'][parameterization_set - 1] # conversion factor from spring flow to fertility rate
        self.muq = paramdf['muq'][parameterization_set - 1]        
        self.sigq = paramdf['sigq'][parameterization_set - 1]        
        self.NHbar = paramdf['NHbar'][parameterization_set - 1]        
        self.Nth = paramdf['Nth'][parameterization_set - 1]        
        self.l = paramdf['l'][parameterization_set - 1]        
        self.p = paramdf['p'][parameterization_set - 1] # Reward for persistence
        self.c = paramdf['c'][parameterization_set - 1] # Cost for augmentation
        self.gamma = paramdf['gamma'][parameterization_set - 1] # Discount factor
        self.extpenalty = paramdf['extpenalty'][parameterization_set - 1] # Penalty for extinction

        # Define state space and action space based on your document
        if discretization_set == 0:
            self.states = {
                "NW": [0,10000000], # Population size, continuous
                "NWm1": [self.Nth, 10000000], # last year population size, continuous
                "NH": [0, 300000], # hatchery population size, continuous
                "H": [0.56, 0.86], # Heterozygosity, continuous
                "q": [65, 848], # Spring Flow, continuous
                "tau": [0, 1]  # 0 for Fall, 1 for Spring, discrete
            }

            self.actions = {
                "a": list(range(0, 300001, 10000)) # stocking/production number, discrete
            }


        self.statespace_dim = [-1,-1,-1,-1,-1,2] # continuous states = -1, discrete states = number of states
        self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))

        # varname idx 
        self.statevaridx = {key: idx for idx, key in enumerate(self.states.keys())}
        self.actionvaridx = {key: idx for idx, key in enumerate(self.actions.keys())}

        # Initialize state
        self.state = self.reset(initstate)

        
    def reset(self, initstate=None):
        if initstate == None:
            initstate = np.ones(len(self.statespace_dim))*-1

        # Initialize state variables
        new_state = []
        if initstate[5] == -1:
            season  = random.choice([0,1])
        else:
            season = initstate[5]
        if season == 0: # spring
            if initstate[0] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["NW"][1] - self.Nth) + self.Nth) # don't start from the smallest population size
            else: 
                new_state.append(initstate[0])
            if initstate[1] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["NWm1"][1] - self.Nth) + self.Nth) # don't start from the smallest population size
            else:
                new_state.append(initstate[1])
            if initstate[2] == -1:
                new_state.append(self.states["NH"][0]) # start from no hatchery fish
            else:
                new_state.append(initstate[2])
            if initstate[3] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["H"][1] - self.states["H"][0]) + self.states["H"][0]) # don't start from the smallest population size
            else: 
                new_state.append(initstate[3])
            if initstate[4] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["q"][1] - self.states["q"][0]) + self.states["q"][0]) # don't start from the smallest population size
            else:
                new_state.append(initstate[4])
        else: # fall
            if initstate[0] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["NW"][1] - self.Nth) + self.Nth) # don't start from the smallest population size
            else:
                new_state.append(initstate[0])
            if initstate[1] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["NWm1"][1] - self.Nth) + self.Nth) # don't start from the smallest population size
            else:
                new_state.append(initstate[1])
            if initstate[2] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["NH"][1] - self.states["NH"][0]) + self.states["NH"][0]) # don't start from the smallest population size
            else:  
                new_state.append(initstate[2])
            if initstate[3] == -1:
                new_state.append(random.uniform(0, 1)*(self.states["H"][1] - self.states["H"][0]) + self.states["H"][0]) # don't start from the smallest population size
            else: 
                new_state.append(initstate[3])
            if initstate[4] == -1:
                new_state.append(self.states["q"][0]) # spring flow state is not relevant in fall
            else:
                new_state.append(initstate[4])

        new_state.append(season)
        self.state = new_state
        return self.state
    
    def step(self, action):
        # Compute next state and reward based on action and transition rules
        
        NW = self.state[0]
        NWm1 = self.state[1]
        NH = self.state[2]
        H = self.state[3]
        q = self.state[4]
        tau = self.state[5]
        a = self.actions["a"][action]
        # Check termination
        if NW > self.Nth:  # Extinction threshold
                    
            # Update season
            tau_next = 1 - tau
            # Transition logic
            if tau == 0:  # Spring-Fall
                F = self._recruitment_rate(q)
                H_next = self._nextgen_heterozygosity(H,NW,NWm1)
                s = self._survival_rate(H_next) # survival rate dependent on the next generation's heterozygosity
                recruitment = F*NW
                try:
                    NW_next = np.random.binomial(np.random.poisson(recruitment),s)
                    #display(f's={s:.2f}, F={F:.2f}, spawner={spawner}, recruitment={recruitment}, poisson used, NW_next={NW_next}')
                except ValueError:
                    NW_next = np.random.binomial(np.ceil(np.random.normal(recruitment, np.sqrt(recruitment))),s)
                    #display(f's={s:.2f}, F={F:.2f}, spawner={spawner}, recruitment={recruitment}, normal used, NW_next={NW_next}')
                NWm1_next = NW
                q_next = 0 #max(np.random.normal(self.muq, self.sigq),0) # q initialized
                NH_next = a
                # Reward
                reward = self.p - self.c if a > 0 else self.p
            else:  # Winter-Spring
                H_next = self._update_heterozygosity(H, NW, a)
                if a <= NH:
                    s = self._survival_rate(H_next) # survival rate dependent on the mixed population's heterozygosity
                    NW_next = np.random.binomial(NW + a, s)
                    reward = self.p
                else: # action is invalid
                    # penalty for invalid action: reward is -p and the episode ends
                    reward = 0
                    NW_next = 0
                    s = 0
                NWm1_next = NWm1 # not updated
                q_next =  max(np.random.normal(self.muq, self.sigq),0)
                NH_next = 0 # all hatchery fish that aren't released are discarded
            # clip the state to the boundaries
            NW_next = min(max(NW_next, self.states["NW"][0]), self.states["NW"][1])
            NWm1_next = min(max(NWm1_next, self.states["NWm1"][0]), self.states["NWm1"][1])
            NH_next = min(max(NH_next, self.states["NH"][0]), self.states["NH"][1])
            H_next = min(max(H_next, self.states["H"][0]), self.states["H"][1])
            q_next = min(max(q_next, self.states["q"][0]), self.states["q"][1])
            # update state
            self.state = [NW_next, NWm1_next, NH_next, H_next, q_next, tau_next]
            # Check termination
            done  = False
        else:
            # extinction
            done = True
            reward = self.extpenalty
            s = 0

        return reward, done, s

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
