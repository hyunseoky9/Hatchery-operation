import numpy as np
from math import floor
import random
from IPython.display import display
import pandas as pd
class Env1_0:
    def __init__(self,initstate,parameterization_set,discretization_set):
        self.envID = 'Env1.0'
        self.partial = False
        self.episodic = True
        self.discset = discretization_set
        self.contstate= False
        # Define state space and action space based on your document
        if discretization_set == 0:
            self.states = {
                "NW": [1000, 3000, 8048, 21590, 57920, 155384, 416848, 1118278, 3000000, 10000000], # Population size
                "NWm1": [1000, 3000, 8048, 21590, 57920, 155384, 416848, 1118278, 3000000, 10000000], # NW minus 1
                "NH": [0, 3333, 66666, 100000, 133333, 166666, 200000, 233333, 266666, 300000], # hatchery population size
                "H": [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86], # Heterozygosity
                "q": [65, 322, 457, 592, 848], # Spring Flow
                "tau": [0, 1]  # 0 for Fall, 1 for Spring
            }

            self.actions = {
                "a": [0, 3333, 66666, 100000, 133333, 166666, 200000, 233333, 266666, 300000]
            }
        elif discretization_set == 1:
            self.states = {
                "NW": [1000, 3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "NWm1": [3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "NH": [0, 75000, 150000, 225000, 300000], # hatchery population size
                "H": [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86], # Heterozygosity
                "q": [65, 322, 457, 592, 848], # Spring Flow
                "tau": [0, 1]  # 0 for Fall, 1 for Spring
            }

            self.actions = {
                "a": [0, 75000, 150000, 225000, 300000]
            }
        elif discretization_set == 2:
            self.states = {
                "NW": [1000, 2500000, 5000000], # Population size
                "NWm1": [2500000, 5000000], # Population size
                "NH": [0, 100000, 200000], # hatchery population size
                "H": [0.56, 0.71, 0.86], # Heterozygosity
                "q": [65, 322, 457], # Spring Flow
                "tau": [0, 1]  # 0 for Fall, 1 for Spring
            }

            self.actions = {
                "a": [0, 100000, 200000]
            }

        self.statespace_dim = list(map(lambda x: len(x[1]), self.states.items()))
        self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
        # IMPORTANT INFO:
        # statespace: 7*6*5*7*5*2 = 14700
        # actionspace: 7*6*5*7*5*2*5 = 73500
        # reachable state space reduced b/c only 1 spring flow state possible when tau=1
        # & only 1 hatchery state possible when tau=0
        # reachable state space: 7*6*1*7*5 + 7*6*5*7*1 = 2940
        # reachable state-action space 7*6*1*7*5*2*(5) = 14700


        # varname idx 
        self.statevaridx = {key: idx for idx, key in enumerate(self.states.keys())}
        self.actionvaridx = {key: idx for idx, key in enumerate(self.actions.keys())}

        

        # Initialize state
        self.state = self.reset(initstate)
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
        
    def reset(self, initstate=None):
        if initstate == None:
            initstate = list(np.ones(len(self.statespace_dim))*-1)

        # Initialize state variables
        new_state = []
        if initstate[5] == -1:
            season  = random.choice([0,1])
        else:
            season = initstate[5]
        if season == 0: # spring
            if initstate[0] == -1:
                new_state.append(random.choice(np.arange(1, len(self.states["NW"])))) # don't start from the smallest population size
            else: 
                new_state.append(initstate[0])
            if initstate[1] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["NWm1"]))))
            else:
                new_state.append(initstate[1])
            if initstate[2] == -1:
                new_state.append(0) # start from no hatchery fish
            else:
                new_state.append(initstate[2])
            if initstate[3] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["H"]))))
            else: 
                new_state.append(initstate[3])
            if initstate[4] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["q"]))))
            else:
                new_state.append(initstate[4])
        else: # fall
            if initstate[0] == -1:
                new_state.append(random.choice(np.arange(1, len(self.states["NW"])))) # don't start from the smallest population size
            else:
                new_state.append(initstate[0])
            if initstate[1] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["NWm1"]))))
            else:
                new_state.append(initstate[1])
            if initstate[2] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["NH"]))))
            else:  
                new_state.append(initstate[2])
            if initstate[3] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["H"]))))
            else: 
                new_state.append(initstate[3])
            if initstate[4] == -1:
                new_state.append(0) # spring flow state is not relevant in fall
            else:
                new_state.append(initstate[4])
        new_state.append(season)
        self.state = new_state
        return self.state
    
    def step(self, action):
        # Compute next state and reward based on action and transition rules
        
        NW = self.states["NW"][self.state[0]]
        NWm1 = self.states["NWm1"][self.state[1]]
        NH = self.states["NH"][self.state[2]]
        H = self.states["H"][self.state[3]]
        q = self.states["q"][self.state[4]]
        tau = self.states["tau"][self.state[5]]
        a = self.actions["a"][action]
        # Check termination
        if NW > self.Nth:  # Extinction threshold
                    
            # Update season
            tau_next = 1 - tau
            # Transition logic
            if tau == 0:  # Spring-Fall (spring)
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
                NH_next = action
                # Reward
                reward = self.p - self.c if a > 0 else self.p

            else:  # Winter-Spring (fall)
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

            # put it into defined discrete states
            NW_next = self._discretize(NW_next, self.states['NW'])
            H_next = self._discretize(H_next, self.states['H'])
            q_next = self._discretize(q_next, self.states['q'])
            # Update state
            NW_next_idx = np.where(np.array(self.states['NW']) == NW_next)[0][0]
            NWm1_next = np.where(np.array(self.states['NWm1']) == NWm1_next)[0][0]
            H_next_idx = np.where(np.array(self.states['H']) == H_next)[0][0]
            q_next_idx = np.where(np.array(self.states['q']) == q_next)[0][0]
            self.state = [NW_next_idx, NWm1_next, NH_next, H_next_idx, q_next_idx, tau_next]
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
    def reachable_states(self):
        # return the reachable state ids
        states = []
        for tau in range(len(self.states['tau'])):
            if tau == 0: # spring
                for NW in range(len(self.states['NW'])):
                    for NWm1 in range(len(self.states['NWm1'])):
                        for H in range(len(self.states['H'])):
                            for q in range(len(self.states['q'])):
                                state = [NW, NWm1, 0, H, q, tau]
                                states.append(self._flatten(state))
            else: # fall
                for NW in range(len(self.states['NW'])):
                    for NWm1 in range(len(self.states['NWm1'])):
                        for NH in range(len(self.states['NH'])):
                            for H in range(len(self.states['H'])):
                                    state = [NW, NWm1, NH, H, 0, tau]
                                    states.append(self._flatten(state))
        return states
    
    def reachable_state_actions(self):
        # return the reachable state-action ids
        # statespace: 7*6*5*7*5*2 = 14700
        # actionspace: 7*6*5*7*5*2*5 = 73500
        # reachable state space reduced b/c only 1 spring flow state possible when tau=1
        # & only 1 hatchery state possible when tau=0
        # reachable state space: 7*6*1*7*5 + 7*6*5*7*1 = 2940
        # reachable state-action space 7*6*1*7*5*2*(5) = 14700

        sapair = []
        for tau in range(len(self.states['tau'])):
            if tau == 0:
                for NW in range(len(self.states['NW'])):
                    for NWm1 in range(len(self.states['NWm1'])):
                        for H in range(len(self.states['H'])):
                            for q in range(len(self.states['q'])):
                                for a in range(len(self.actions['a'])):
                                    state = [NW, NWm1, 0, H, q, tau]
                                    sapair.append((self._flatten(state), a))
            else:
                for NW in range(len(self.states['NW'])):
                    for NWm1 in range(len(self.states['NWm1'])):
                        for NH in range(len(self.states['NH'])):
                            for H in range(len(self.states['H'])):
                                for a in range(len(self.actions['a'])):
                                        state = [NW, NWm1, NH, H, 0, tau]
                                        sapair.append((self._flatten(state), a))
        return sapair
    
    def Qsubcalc(self, Q, ranges):
        #, NWrange, NWm1range, NHrange, Hrange, qrange, taurange
        # return part of the high dimensional Q
        # returns a two dimensional matrix of Q fixing the other state variables 
        # if list is given in any of the state variables, that state variable will be in the matrix
        # all the other scalar values for the ranges indicate that the variable is fixed.
        axis = []
        axisstate = []
        nonaxisstate = []
        nonaxisstateval = []
        for i in range(len(ranges)):
            if isinstance(ranges[i], np.ndarray):
                axis.append(ranges[i])
                axisstate.append(i)
            else:
                nonaxisstate.append(i)
                nonaxisstateval.append(ranges[i])

        if len(axis) > 2:
            display('too many ranges!')
            return 0
        Qsub = [np.zeros([len(axis[0]), len(axis[1])]) for i in range(self.actionspace_dim[0])]
        for a in range(len(self.actions['a'])):
            for i in range(len(axis[0])):
                for j in range(len(axis[1])):
                    state = np.zeros(len(self.statespace_dim))
                    state[axisstate[0]] = i
                    state[axisstate[1]] = j
                    for k in range(len(nonaxisstate)):
                        state[nonaxisstate[k]] = nonaxisstateval[k]
                    stateid = int(self._flatten(state))
                    Qsub[a][i,j] = Q[stateid,a]

        return Qsub
