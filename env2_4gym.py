import numpy as np
from math import floor
import random
from IPython.display import display
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class Env2_4gym(gym.Env):
    """
    Same as Env2.0 but catch (y) is observed every season (both in fall and spring)
    """
    def __init__(self,config=None):
        super(Env2_4gym, self).__init__()
        # Load configuration
        if config is None:
            config = {}
        # Safe defaults
        initstate = config.get("initstate", [-1, -1, -1, -1, -1, -1])
        parameterization_set = config.get("parameterization_set", 2)
        discretization_set = config.get("discretization_set", 0)

        self.envID = 'Env2.4'
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
                "y": [0, 15, 30, 45], # observed catch from fall monitoring. -1= no observed catch (for spring); 45 is actually anything gretaer than 45
                "ONH": [0, 75000, 150000, 225000, 300000], # observed hatchery fish
                "OH": [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86], # observed heterozygosity
                "Oq": [65, 322, 457, 592, 848], # observed spring flow
                "Otau": [0, 1]  # observed season
            }
            self.actions = {
                "a": [0, 75000, 150000, 225000, 300000]
            }
        elif discretization_set == 1:
            self.states = {
                "NW": [1000, 2500000, 5000000], # Population size
                "NWm1": [2500000, 5000000], # Population size
                "NH": [0, 100000, 200000], # hatchery population size
                "H": [0.56, 0.71, 0.86], # Heterozygosity
                "q": [65, 322, 457], # Spring Flow
                "tau": [0, 1]  # 0 for Fall, 1 for Spring
            }
            self.observations = {
                "y": [-1, 0, 30], # observed catch from fall monitoring. -1= no observed catch (for spring); 45 is actually anything gretaer than 45
                "ONH": [0, 100000, 200000], # hatchery population size
                "OH": [0.56, 0.71, 0.86], # Heterozygosity
                "Oq": [65, 322, 457], # Spring Flow
                "Otau": [0, 1]  # 0 for Fall, 1 for Spring
            }
            self.actions = {
                "a": [0, 100000, 200000]
            }
        elif discretization_set == 2:
            self.states = {
                "NW": [1000, 3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "NWm1": [3000, 15195, 76961, 389806, 1974350, 10000000], # Population size
                "NH": [0, 75000, 150000, 225000, 300000], # hatchery population size
                "H": [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86], # Heterozygosity
                "q": [65, 322, 457, 592, 848], # Spring Flow
                "tau": [0, 1]  # 0 for Fall, 1 for Spring
            }
            self.observations = {
                "y": [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46], # observed catch from fall monitoring. -1= no observed catch (for spring); 45 is actually anything gretaer than 45
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
        self.low = np.zeros(5, dtype=np.float32)  # each dimension min=0
        self.high = np.array([dim_size - 1 for dim_size in self.obsspace_dim], dtype=np.float32)  # each dimension max=(size-1)
        self.observation_space = spaces.Box(low=self.low, high=self.high, shape=(5,), dtype=np.float32)
        #self.observation_space = spaces.MultiDiscrete([len(v) for v in self.states.values()])
        self.action_space = spaces.Discrete(len(self.actions["a"]))
        
        # Define parameters
        # call in parameterization dataset csv
        # Read csv 'parameterization_env1.0.csv'
        self.parset = parameterization_set - 1
        parameterization_set_filename = 'parameterization_env2.0.csv'
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
        self.theta = paramdf['theta'][parameterization_set - 1] # detection probability
        self.sigy = paramdf['sigy'][parameterization_set - 1] # overdispersion parameter for observed catch


        # Specify if you want to support or define a maximum number of steps.
        self.max_steps = 100
        self.current_step = 0
        self.absorbing_counter = 0

        # Initialize state and observation
        self.state, self.obs = self.reset(initstate)


    def reset(self, initstate=None, seed=None, options=None):
        self.current_step = 0
        self.absorbing_counter = 0
        if initstate is None:
            initstate = [-1, -1, -1, -1, -1, -1]
        # Initialize state variables
        new_state = []
        new_obs = []
        if initstate[5] == -1:
            season  = random.choice([0,1])
        else:
            season = initstate[5]
        if season == 0: # spring
            if initstate[0] == -1:
                new_state.append(random.choice(np.arange(1, len(self.states["NW"])))) # don't start from the smallest population size
            else:
                new_state.append(initstate[0])

            new_y = self._fallmonitoring(self.states["NW"][new_state[0]])
            new_y = self._discretize(new_y, self.observations['y'])
            new_y = np.where(np.array(self.observations['y']) == new_y)[0][0]
            new_obs.append(new_y) # observed catch in fall

            if initstate[1] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["NWm1"]))))
            else:
                new_state.append(initstate[1])
            if initstate[2] == -1:
                new_state.append(0) # start from no hatchery fish
                new_obs.append(0)
            else:
                new_state.append(initstate[2])
                new_obs.append(initstate[2])
            if initstate[3] == -1:
                idx3 = random.choice(np.arange(0, len(self.states["H"])))
                new_state.append(idx3)
                new_obs.append(idx3)
            else: 
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            if initstate[4] == -1:
                idx4 = random.choice(np.arange(0, len(self.states["q"])))
                new_state.append(idx4)
                new_obs.append(idx4)
            else:
                new_state.append(initstate[4])
                new_obs.append(initstate[4])
        else: # fall

            if initstate[0] == -1:
                new_state.append(random.choice(np.arange(1, len(self.states["NW"])))) # don't start from the smallest population size
            else:
                new_state.append(initstate[0])

            new_y = self._fallmonitoring(self.states["NW"][new_state[0]])
            new_y = self._discretize(new_y, self.observations['y'])
            new_y = np.where(np.array(self.observations['y']) == new_y)[0][0]
            new_obs.append(new_y) # observed catch in fall
            if initstate[1] == -1:
                new_state.append(random.choice(np.arange(0, len(self.states["NWm1"]))))
            else:
                new_state.append(initstate[1])
            if initstate[2] == -1:
                idx2 = random.choice(np.arange(0, len(self.states["NH"])))
                new_state.append(idx2)
                new_obs.append(idx2)
            else:  
                new_state.append(initstate[2])
                new_obs.append(initstate[2])
            if initstate[3] == -1:
                idx3 = random.choice(np.arange(0, len(self.states["H"])))
                new_state.append(idx3)
                new_obs.append(idx3)
            else: 
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            if initstate[4] == -1:
                new_state.append(0) # spring flow state is not relevant in fall
                new_obs.append(0) 
            else:
                new_state.append(initstate[4])
                new_obs.append(initstate[4])
        new_state.append(season)
        new_obs.append(season)
        self.state = new_state
        self.obs = new_obs
        return np.array(self.obs), {'true_state': np.array(self.state)}
    
    def step(self, action):
        # Compute next state and reward based on action and transition rules

        truncated = False
        done = False 
        
        NW = self.states["NW"][self.state[0]]
        NWm1 = self.states["NWm1"][self.state[1]]
        NH = self.states["NH"][self.state[2]]
        H = self.states["H"][self.state[3]]
        q = self.states["q"][self.state[4]]
        tau = self.states["tau"][self.state[5]]
        a = self.actions["a"][action]
        # Check termination
        # Update season
        tau_next = 1 - tau
        if NW > self.Nth:  # Extinction threshold
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
                    invalidcost = 0
                else: # action is invalid
                    # penalty for invalid action: reward is -p and the episode ends
                    NW_next = 0
                    s = 0
                    invalidcost = 0
                NWm1_next = NWm1 # not updated
                q_next =  max(np.random.normal(self.muq, self.sigq),0)
                NH_next = 0 # all hatchery fish that aren't released are discarded
                # Reward
                reward = self.p + invalidcost

            # put it into defined discrete states
            NW_next = self._discretize(NW_next, self.states['NW'])
            H_next = self._discretize(H_next, self.states['H'])
            q_next = self._discretize(q_next, self.states['q'])

            # observation is based on monitoring (monitoring made in both fall and spring)
            y_next = self._fallmonitoring(NW_next)
            y_next = self._discretize(y_next, self.observations['y'])
            
            # Update state
            NW_next_idx = np.where(np.array(self.states['NW']) == NW_next)[0][0]
            NWm1_next = np.where(np.array(self.states['NWm1']) == NWm1_next)[0][0]
            H_next_idx = np.where(np.array(self.states['H']) == H_next)[0][0]
            q_next_idx = np.where(np.array(self.states['q']) == q_next)[0][0]
            y_next_idx = np.where(np.array(self.observations['y']) == y_next)[0][0]
            self.state = [NW_next_idx, NWm1_next, NH_next, H_next_idx, q_next_idx, tau_next]
            self.obs = [y_next_idx, NH_next, H_next_idx, q_next_idx, tau_next]
            # Check termination
            done  = False
        else: # extinct.
            # Transition logic
            if tau == 0:  # Spring-Fall (spring)
                q_next = self.states['q'][0] #max(np.random.normal(self.muq, self.sigq),0) # q initialized
                NH_next = action
                reward = self.extpenalty - self.c if a > 0 else self.extpenalty
            else:  # Winter-Spring (fall)
                q_next =  max(np.random.normal(self.muq, self.sigq),0)
                q_next = self._discretize(q_next, self.states['q'])

                NH_next = 0 # all hatchery fish that aren't released are discarded
                reward = self.extpenalty

            y_next = 0 # 0 catch when extinct.
            
            # Update state
            NW_next_idx = 0
            NWm1_next = 0
            H_next_idx = 0
            q_next_idx = np.where(np.array(self.states['q']) == q_next)[0][0]
            y_next_idx = np.where(np.array(self.observations['y']) == y_next)[0][0]
            self.state = [NW_next_idx, NWm1_next, NH_next, H_next_idx, q_next_idx, tau_next]
            self.obs = [y_next_idx, NH_next, H_next_idx, q_next_idx, tau_next]
            # extinction
            done = False
            s = 0
        # Update step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        if self.state[0] == 0:
            self.absorbing_counter += 1
        if self.absorbing_counter > 3:
            truncated = True

        return np.array(self.obs), reward, done, truncated,{'s': s, 'true_state': np.array(self.state)}

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
    
    def observation_prob(self):
        sample = 10000
        y_prob = np.zeros([len(self.states['NW']),len(self.observations['y'])])
        for i in range(len(self.states['NW'])):
            for _ in range(sample):
                y_next = self._fallmonitoring(self.states['NW'][i])
                y_next = self._discretize(y_next, self.observations['y'])
                y_next_idx = np.where(np.array(self.observations['y']) == y_next)[0][0]
                y_prob[i,y_next_idx] += 1
            y_prob[i] /= sample
            print(f'NW[{i}] done')
        return y_prob
    
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
