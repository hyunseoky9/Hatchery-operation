import multiprocessing
import logging
from joblib import Parallel, delayed
import time
import itertools
import pickle
from env1_0 import Env1_0
from env0_0 import Env0_0
import numpy as np
from scipy.stats import norm, beta, binom, poisson
import scipy.stats as stats
from IPython.display import display
from math import floor

class value_iteration(Env1_0):

    def value_iter(self):
        # do value iteration on the model to get the value function and policy

        # read transition probability
        filename = f"trp{self.envID}_par{self.parset}_dis{self.discset}.pkl"
        display(filename)
        with open(filename, "rb") as file:
            transition_prob = pickle.load(file)
        
        #numstates = len(self.states["NW"])*len(self.states["NWm1"])*len(self.states["NH"])*len(self.states["H"])*len(self.states["q"])*len(self.states["tau"])
        numstates = np.prod(self.statespace_dim)

        # initialize value function
        V = np.zeros(numstates)
        # initialize policy
        policy = np.zeros(numstates)
        # initialize threshold
        if self.envID == 'Env1.0':
            theta = 0.001
        elif self.envID == 'Env0.0':
            theta = 0.00001


        # initialize delta
        delta = 1
        # initialize iteration counter
        iteration = 0
        # measure time
        start = time.time()
        while delta > theta:
            delta = 0
            for s in range(numstates):
                v = V[s]
                v_peraction = []
                for a in range(len(self.actions["a"])): # for each action
                    #print(f'state={self._unflatten(s)}, action{a}')
                    #print(f'transitions = {transition_prob[s][a]}, len={len(transition_prob[s][a])}')
                    if transition_prob[s][a] == (): # termination state-action pairs
                        v_peraction.append(self.reward(s,a))
                    else:
                        v_peraction.append(sum([p*(self.reward(s,a) + self.gamma*V[s1]) for s1,p in transition_prob[s][a]]))
                    #print(v_peraction)
                V[s] = max(v_peraction)
                delta = max(delta,abs(v - V[s]))
            print(f"delta={delta}")
            iteration += 1

            #matrix = np.round(V,2).reshape(self.statespace_dim).transpose()
            #print(matrix)
            #input()
            if iteration % 10 == 0:
                print(f"iteration={iteration}")
        
       # get corresponding policy from the final value function
       # for each state, find the action that maximizes the value function
        for s in range(numstates):
            v_peraction = []
            for a in range(len(self.actions["a"])):
                if transition_prob[s][a] == ():
                    v_peraction.append(self.reward(s,a))
                else:
                    v_peraction.append(sum([p*(self.reward(s,a) + self.gamma*V[s1]) for s1,p in transition_prob[s][a]]))
            policy[s] = np.argmax(v_peraction)

        # get Q function from value function
        Q = self._make_Q(V,transition_prob)
        
        finish = time.time()
        # save value function and policy as pickle
        wd = './value iter results'
        # value function save
        with open(f"{wd}/V_{self.envID}_par{self.parset}_dis{self.discset}_valiter.pkl", "wb") as file:
            pickle.dump(V, file)
        # Q function save
        with open(f"{wd}/Q_{self.envID}_par{self.parset}_dis{self.discset}_valiter.pkl", "wb") as file:
            pickle.dump(Q, file)
        # policy save
        with open(f"{wd}/policy_{self.envID}_par{self.parset}_dis{self.discset}_valiter.pkl", "wb") as file:
            pickle.dump(policy, file)
        print(f"Time taken: {(finish-start)/60} minutes")
        return Q, V, policy

    def compute_transition_probability(self):
        # compute transition probability for each state-action pair
        # output: 2 dimensional list, where 1st dimension is the current state and 2nd is the action.
        # Each coordinate is a nx2 matrix, where n is the number of possible next states, and the 2 
        # columns are the next state and the probability of transitioning to that state.
        
        statespace_dim = self.statespace_dim
        actionspace_dim = self.actionspace_dim
        statespace_size = int(np.array(statespace_dim).prod())
        actionspace_size = int(np.array(actionspace_dim).prod())
        # probability of spring flow q constant so set it here now (env.1.0)
        if self.envID == 'Env1.0':
            q_prob = self._qprobcalc()
            #with open(f"NWprob(tau0)_Env1.0_par1_dis{self.discset}.pkl", "rb") as file:
            #    NWt0prob_lookuptable = pickle.load(file)
            #with open(f"NWprob(tau1)_Env1.0_par1_dis{self.discset}.pkl", "rb") as file:
            #    NWt1prob_lookuptable = pickle.load(file)
            #with open(f"Hprob(tau0)_Env1.0_par1_dis{self.discset}.pkl", "rb") as file:
            #    Ht0prob_lookuptable = pickle.load(file)
            #with open(f"Hprob(tau1)_Env1.0_par1_dis{self.discset}.pkl", "rb") as file:
            #    Ht1prob_lookuptable = pickle.load(file)
        transition_prob = [[0 for i in range(actionspace_size)] for j in range(statespace_size)]
        #measure time
        start = time.time()
        for i in range(statespace_size):
            for j in range(actionspace_size):
                state_idx = self._unflatten(i)
                terminal = self.terminal_state(state_idx)
                if terminal == False:
                    if self.envID == 'Env1.0':
                        transition_prob[i][j] = self._numerical_transition_probcalc(state_idx,j)
                        #transition_prob[i][j] = self.transition_Env1_0(state_idx,j,q_prob,
                        #    NWt0prob_lookuptable,NWt1prob_lookuptable,Ht0prob_lookuptable,Ht1prob_lookuptable)
                    if self.envID == 'Env0.0':
                        transition_prob[i][j] = self._numerical_transition_probcalc(state_idx,j)
                else:
                    transition_prob[i][j] = ()
                # print progress every 1000th iteration of i
            if i % 5 == 0:
                print(f"i={i}/{statespace_size}")

        finish = time.time()
        # print time in minutes
        print(f"Time taken: {(finish-start)/60} minutes")
        
        # save transitionprobability as pickle
        with open(f"transiti    on_prob{self.envID}_par{self.parset}_dis{self.discset}.pkl", "wb") as file:
            pickle.dump(transition_prob, file)
        
        return transition_prob
    
    def transition_Env1_0(self, state_index, action_idx, q_prob, NWt0prob_lookuptable, NWt1prob_lookuptable, Ht0prob_lookuptable, Ht1prob_lookuptable):
        # Compute next state and reward based on action and transition rules
        NW = self.states["NW"][state_index[0]]  
        NWm1 = self.states["NWm1"][state_index[1]]
        NH = self.states["NH"][state_index[2]]
        H = self.states["H"][state_index[3]]
        q = self.states["q"][state_index[4]]
        tau = self.states["tau"][state_index[5]]
        a = self.actions["a"][action_idx]
        
        # Check termination
        if (tau == 1 and a > NH) or (NW <= self.Nth):
            stateprob = () # empty tuple
        else:
            tau_prob = np.zeros(len(self.states["tau"]))
            NW_prob = np.zeros(len(self.states["NW"]))
            NWm1_prob = np.zeros(len(self.states["NWm1"]))
            NH_prob = np.zeros(len(self.states["NH"]))
            H_prob = np.zeros(len(self.states["H"]))
            
            if tau == 0:
                tau_prob[1] = 1
                NW_prob = NWt0prob_lookuptable[state_index[0],state_index[1],state_index[3],state_index[4]]
                NWm1_prob[state_index[0]-1] = 1
                NH_prob[action_idx] = 1
                H_prob = Ht0prob_lookuptable[state_index[0], state_index[1],state_index[3]]
                q_prob = np.zeros(len(self.states["q"]))
                q_prob[0] = 1
            else:
                tau_prob[0] = 1
                NW_prob = NWt1prob_lookuptable[state_index[0],state_index[3],action_idx] # self._NWprobcalc_t1(NW,a,H)
                NWm1_prob[state_index[1]] = 1
                NH_prob[0] = 1
                H_prob = Ht1prob_lookuptable[state_index[0],state_index[3],action_idx] #
            NW_available_states = np.where(np.array(NW_prob) > 0)[0]
            NWm1_available_states = np.where(np.array(NWm1_prob) > 0)[0]
            NH_available_states = np.where(np.array(NH_prob) > 0)[0]
            H_available_states = np.where(np.array(H_prob) > 0)[0]
            q_available_states = np.where(np.array(q_prob) > 0)[0]
            tau_available_states = np.where(np.array(tau_prob) > 0)[0]
            

            available_states = list(itertools.product(NW_available_states,NWm1_available_states,
                                                NH_available_states,H_available_states,
                                                q_available_states,tau_available_states))
            
            stateprob = []
            probsum = 0
            for state in available_states:
                prob = NW_prob[state[0]]*NWm1_prob[state[1]]*NH_prob[state[2]]\
                    *H_prob[state[3]]*q_prob[state[4]]*tau_prob[state[5]]
                if prob < 0:
                    print(f"one of the prob is negative: {(prob)}")
                    print(f"current state: {state_index}")
                    print(f"next state: {state}")
                if prob < 10**-5:
                    prob = 0
                    continue
                stateid = self._flatten(state)
                stateprob.append((stateid,prob))
                probsum += prob
            stateprob = tuple(stateprob)
            if probsum < 0.99 or probsum > 1.01:
                print(f"sum of probabilities is not 1: {probsum}")
                print(f"current state: {state_index}")
                print(f"action: {action_idx}")
        return stateprob # tuple of (stateid,probability) pair tuples

    def _numerical_transition_probcalc(self,state_idx,action_idx):
        # numerically derive all transition probability
        # can be done for all environment models technically. made for env0.0 first
        samplenum = 30000
        new_states = []
        for sample in range(samplenum):
            self.reset(state_idx)
            foo = self.step(action_idx)
            new_state = self._flatten(self.state)
            new_states.append(new_state)
        uniques, counts = np.unique(new_states, return_counts=True)
        prob = counts/np.sum(counts)
        stateprob = [(uniques[i],prob[i]) for i in range(len(uniques))]
        stateprob = tuple(stateprob)
        return stateprob


    def _NWprobcalc0(self):
        # numerically derive transition probability for NW at tau=0(spring)
        # for env1.0
        samplenum = 1000#10000
        # for every H, NWm1, and q, calculate the probability of NW.
        NWprob0 = np.zeros([len(self.states["NW"]),len(self.states["NWm1"]),len(self.states["H"]),len(self.states["q"]),len(self.states["NW"])]) # probability of next NW states when tau=0
        for i in range(len(self.states["NW"])):
            if i == 0: # termination state
                continue
            for j in range(len(self.states["NWm1"])):
                for k in range(len(self.states["H"])):
                    for l in range(len(self.states["q"])):
                        NW = self.states["NW"][i]
                        NWm1 = self.states["NWm1"][j]
                        H = self.states["H"][k]
                        q = self.states["q"][l]
                        F = self._recruitment_rate(q)
                        for sample in range(samplenum):
                            H_next = self._nextgen_heterozygosity(H,NW,NWm1)
                            s = self._survival_rate(H_next) # survival rate dependent on the next generation's heterozygosity
                            recruitment = F*NW
                            try:
                                NW_next = np.random.binomial(np.random.poisson(recruitment),s)
                            except ValueError:
                                NW_next = np.random.binomial(np.ceil(np.random.normal(recruitment, np.sqrt(recruitment))),s)
                            NW_next = self._discretize(NW_next, self.states['NW'])
                            NW_next_idx = np.where(np.array(self.states['NW']) == NW_next)[0][0]
                            NWprob0[i,j,k,l,NW_next_idx] += 1
                        NWprob0[i,j,k,l,:] = NWprob0[i,j,k,l,:]/samplenum                    
            print(f"nwt0 i={i}")
        with open(f"NWprob(tau0)_{self.envID}_par{self.parset}_dis{self.discset}.pkl", "wb") as file:
            pickle.dump(NWprob0, file)
        return NWprob0

    def _NWprobcalc1(self):
        # for env1.0
        # numerically derive transition probability for NW at tau=1(fall)
        samplenum = 100000#100000
        # for every H, NWm1, and q, calculate the probability of NW.
        NWprob1 = np.zeros([len(self.states["NW"]),len(self.states["H"]),len(self.actions['a']),len(self.states["NW"])]) # probability of next NW states when tau=0
        for i in range(len(self.states["NW"])):
            if i == 0: # termination state
                continue
            for j in range(len(self.states["H"])):
                for k in range(len(self.actions["a"])):
                    NW = self.states["NW"][i]
                    H = self.states["H"][j]
                    a = self.actions["a"][k]
                    for sample in range(samplenum):
                        H_next = self._update_heterozygosity(H, NW, a)
                        s = self._survival_rate(H_next) # survival rate dependent on the next generation's heterozygosity
                        NW_next = np.random.binomial(NW + a, s)
                        NW_next = self._discretize(NW_next, self.states['NW'])
                        NW_next_idx = np.where(np.array(self.states['NW']) == NW_next)[0][0]
                        NWprob1[i,j,k,NW_next_idx] += 1
                    NWprob1[i,j,k,:] = NWprob1[i,j,k,:]/samplenum
            print(f"nwt1 i={i}")

        with open(f"NWprob(tau1)_{self.envID}_par{self.parset}_dis{self.discset}.pkl", "wb") as file:
            pickle.dump(NWprob1, file)
        return NWprob1

    def _Hprobcalc0(self):
        # for env1.0
        # numerically derive transition probability for heterozygosity at tau=0(spring)
        samplenum = 100000#100000
        Hprob0 = np.zeros([len(self.states["NW"]),len(self.states["NWm1"]),len(self.states["H"]),len(self.states["H"])])
        for i in range(len(self.states["NW"])):
            if i == 0:
                continue
            for j in range(len(self.states["NWm1"])):
                for k in range(len(self.states["H"])):
                    NW = self.states["NW"][i]
                    NWm1 = self.states["NWm1"][j]
                    H = self.states["H"][k]
                    for sample in range(samplenum):
                        H_next = self._nextgen_heterozygosity(H,NW,NWm1)
                        H_next = self._discretize(H_next,self.states['H'])
                        H_next_idx = np.where(np.array(self.states['H']) == H_next)[0][0]
                        Hprob0[i,j,k,H_next_idx] += 1
                    Hprob0[i,j,k,:] = Hprob0[i,j,k,:]/samplenum
            print(f"ht0 i={i}")

        with open(f"Hprob(tau0)_{self.envID}_par{self.parset}_dis{self.discset}.pkl", "wb") as file:
            pickle.dump(Hprob0, file)
        return Hprob0

    def _Hprobcalc1(self):
        # for env1.0
        # numerically derive transition probability for heterozygosity at tau=1(fall)
        samplenum = 10000#100000
        Hprob1 = np.zeros([len(self.states["NW"]),len(self.states["H"]),len(self.actions['a']),len(self.states["H"])])
        for i in range(len(self.states["NW"])):
            if i == 0:
                continue
            for j in range(len(self.states["H"])):
                for k in range(len(self.actions["a"])):
                    NW = self.states["NW"][i]
                    H = self.states["H"][j]
                    a = self.actions["a"][k]
                    for sample in range(samplenum):
                        H_next = self._update_heterozygosity(H,NW,a)
                        H_next = self._discretize(H_next,self.states['H'])
                        H_next_idx = np.where(np.array(self.states['H']) == H_next)[0][0]
                        Hprob1[i,j,k,H_next_idx] += 1
                    Hprob1[i,j,k,:] = Hprob1[i,j,k,:]/samplenum
            print(f"ht1i={i}")

        with open(f"Hprob(tau1)_{self.envID}_par{self.parset}_dis{self.discset}.pkl", "wb") as file:
            pickle.dump(Hprob1, file)
        return Hprob1

        
    def _qprobcalc(self):
        # for env1.0
        # numerically derive transition probability for spring flow q
        samplenum = 100000
        q_prob = np.zeros(len(self.states["q"]))
        for i in range(len(self.states["q"])):
            q = self.states["q"][i]
            for sample in range(samplenum):
                q_next = max(np.random.normal(self.muq, self.sigq),0)
                q_next = self._discretize(q_next,self.states['q'])
                q_next_idx = np.where(np.array(self.states['q']) == q_next)[0][0]
                q_prob[q_next_idx] += 1
            q_prob = q_prob/samplenum
        return q_prob
        #q_prob = self._stochastic_discretized_prob(self.states["q"],'normal',[self.muq,self.sigq])
        #return q_prob

    # DEPRECATED
    #def _Hprobcalc_t0(self,H,NW,NWm1):
    #    # calculate probability of each H given NW and NWm1 (reduction in Heterozygosity from population size changes)
    #    Ne = 2/(1/NW + 1/NWm1)
    #    mode = H * (1 + (self.mu - 1 / (2 * Ne)))
    #    a = mode*(self.kappa - 2) + 1
    #    b = (1 - mode)*(self.kappa - 2) + 1
    #    display(f'a={a}, b={b}')
    #    H_prob = self._stochastic_discretized_prob(self.states["H"],'beta',[a,b])
    #    return H_prob
    
    # DEPRECATED
    #def _Hprobcalc_t1(self,H,NW,a):
    #    # calculate probability of H given NW and a (reduction in Heterozygosity from stocking)
    #    H_next = (a * H * (1 - self.l) + NW * H) / (a + NW)
    #    H_prob = self._deterministic_discretized_prob(H_next, self.states["H"])
    #    return H_prob
    
    # DEPRECATED
    #def _NWprobcalc_t0(self,NW,H,q,NWm1):
    #    F = self._recruitment_rate(q)
    #    s = self._survival_rate(H)
    #    NW_prob = self._stochastic_discretized_prob(self.states["NW"],'binom-poisson',[F, NW, s])
    #    return NW_prob

    # DEPRECATED    
    #def _NWprobcalc_t1(self,NW,a,H):
    #    H_next = (a * H * (1 - self.l) + NW * H) / (a + NW) # weighted average of hatchery and wild heterozygosity
    #    s = self._survival_rate(H_next)
    #    NW_prob = self._stochastic_discretized_prob(self.states["NW"],'binom',[NW + a, s])
    #    return NW_prob
    
    # DEPRECATED
    #def _stochastic_discretized_prob(self,possiblestates,disttype,distparams):
    #    # discretize continuous probability distribution
    #    # input: possible states (list of floats)
    #    # output: probability distribution (list of floats)
    #    possiblestates = np.array(possiblestates)
    #    midpoints = (np.array(possiblestates[1:]) - np.array(possiblestates[0:-1]))/2 + np.array(possiblestates[0:-1])
    #    if disttype == 'binom':
    #        midpoints = np.round(midpoints)
    #    if disttype == 'normal':
    #        cdf = norm.cdf(midpoints,distparams[0],distparams[1])
    #    elif disttype == 'beta':
    #        cdf = beta.cdf(midpoints,distparams[0],distparams[1])
    #    elif disttype == 'binom':
    #        cdf = binom.cdf(midpoints,distparams[0],distparams[1])
    #    elif disttype == 'binom-poisson':
    #        ns = floor(distparams[1]*distparams[2])
    #        lrange,urange = stats.binom.ppf([0.001,0.999], distparams[1], distparams[2]) # range of binomial values around mean that will cover 99.9% of the distribution
    #        binomvals_around_mean = np.arange(lrange, urange + 1)
    #        binomP = binom.pmf(binomvals_around_mean,distparams[1],distparams[2]) # calculate binomial pmf for values around mean (n*p)
    #        cdf = []
    #        recruitment = binomvals_around_mean*distparams[0]
    #        for i in range(len(midpoints)):
    #            cdfpre = poisson.cdf(midpoints[i],recruitment)
    #            cdf.append(np.sum(cdfpre*binomP))
    #
    #    prob = np.zeros(len(possiblestates))
    #    for i in range(len(possiblestates)):
    #        if i == 0:
    #            prob[i] = cdf[0]
    #        elif i == len(possiblestates) - 1:
    #            prob[i] = 1 - cdf[-1]
    #        else:
    #            prob[i] = cdf[i] - cdf[i-1]
    #    # change probbabilities smaller than 10^-2 to 0
    #    # prob[prob < 10**] = 0
    #    # make sure it sums to 1.
    #    prob /= np.sum(prob)
    #    return prob

    # DEPRECATED
    #def _deterministic_discretized_prob(self, x, possible_states):
    #    # when the process is deterministic
    #    if x < possible_states[0]:
    #        prob = np.zeros(len(possible_states))
    #        prob[0] = 1
    #    elif x > possible_states[-1]:
    #        prob = np.zeros(len(possible_states))
    #        prob[-1] = 1
    #    else: 
    #        two_nearest_idx = np.argsort(abs(np.array(possible_states) - x))[:2]
    #        lower, upper = np.array(possible_states)[two_nearest_idx]
    #        weights = np.array([upper - x, x - lower])/(upper - lower)  
    #        prob = np.zeros(len(possible_states))
    #        prob[two_nearest_idx[0]] = weights[0]
    #        prob[two_nearest_idx[1]] = weights[1]
    #    return prob

    def reward(self,s_index,a_idx):
        if self.envID == 'Env1.0':
            # compute reward based on state and action
            state = self._unflatten(s_index)
            NW = self.states["NW"][state[0]]
            NH = self.states["NH"][state[2]]
            tau = self.states["tau"][state[5]]
            a = self.actions["a"][a_idx]
            if NW > self.Nth:
                if tau == 0:
                    R = self.p - self.c if a > 0 else self.p
                else:
                    if a <= NH:
                        R = self.p
                    else:
                        R = 0
            else:
                R = self.extpenalty
            return R
        elif self.envID == 'Env0.0':
            # compute reward based on state and action
            state = self._unflatten(s_index)
            NW = self.states["NW"][state[0]]
            a = self.actions["a"][a_idx]
            if NW > self.Nth:
                R = self.p - self.c if a > 0 else self.p
            else:
                R = self.extpenalty
            return R

    def terminal_state(self,state):
        if self.envID == 'Env1.0':
            if state[0] == 0:
                return True
            else:
                return False
        elif self.envID == 'Env0.0':
            if state[0] == 0:
                return True
            else:
                return False
        

    def _make_Q(self, V, transition_prob):
        # get Q function from value function
        numstates = np.prod(self.statespace_dim)
        Q = np.zeros((numstates,len(self.actions["a"])))
        for s in range(numstates):
            for a in range(len(self.actions["a"])):
                if transition_prob[s][a] == ():
                    Q[s,a] = self.reward(s,a)
                else:
                    Q[s,a] = sum([p*(self.reward(s,a) + self.gamma*V[s1]) for s1,p in transition_prob[s][a]])
        return Q



    # get Q function from value function

    # DEPRECATED (TRANSITION PROBABILITY IS TOO LARGE TO ALLOCATE TO EACH CORE)
    #def value_iter_parallel(self):
    #    
    #        # read transition probability
    #        if self.envID == 'Env1.0':
    #            with open(f"transition_prob{self.envID}.pkl", "rb") as file:
    #                transition_prob = pickle.load(file)
    #        
    #        numstates = len(self.states["NW"])*len(self.states["NWm1"])*len(self.states["NH"])*len(self.states["H"])*len(self.states["q"])*len(self.states["tau"])
    #        # initialize value function
    #        V = np.zeros(numstates)
    #        # initialize policy
    #        policy = np.zeros(numstates)
    #        # initialize threshold
    #        theta = 0.01
    #        # initialize delta
    #        delta = 1
    #        # initialize iteration counter
    #        iteration = 0
    #        # measure time
    #        start = time.time()
    #        while delta > theta:
    #            delta = 0
    #            V_new = Parallel(n_jobs=-1)(  # Use all available cores
    #                delayed(value_update_parallel)(
    #                    s, V, transition_prob, self.reward, self.gamma,self.actions
    #                ) for s in range(numstates)
    #            )
    #            delta = np.max(np.abs(V - np.array(V_new[s]))) # compute delta
    #            V = np.array(V_new) # update value function
    #            print(f"delta={delta}")
    #            iteration += 1
    #            if iteration % 10 == 0:
    #                print(f"iteration={iteration}")
    #        
    #    # get corresponding policy from the final value function
    #    # for each state, find the action that maximizes the value function
    #        for s in range(numstates):
    #            v_peraction = []
    #            for a in range(len(self.actions["a"])):
    #                if transition_prob[s][a] == ():
    #                    v_peraction.append(0)
    #                else:
    #                    v_peraction.append(sum([p*(self.reward(s,a) + self.gamma*V[s1]) for s1,p in transition_prob[s][a]]))
    #            policy[s] = np.argmax(v_peraction) 
    #        
    #        finish = time.time()
    #        # save value function and policy as pickle
    #        with open(f"V_{v.envID}.pkl", "wb") as file:
    #            pickle.dump(V, file)
    #        with open(f"policy_{v.envID}.pkl", "wb") as file:
    #            pickle.dump(policy, file)
    #        
    #        print(f"Time taken: {(finish-start)/60} minutes")
    #        return V, policy

    # DEPRECATED (TRANSITION PROBABILITY IS TOO LARGE TO ALLOCATE TO EACH CORE)
    #def value_update(s, V, transition_prob, reward, gamma, actions):
    #    v_peraction = []
    #    for a in range(len(actions["a"])):  # Loop over actions
    #        if transition_prob[s][a] == ():  # Termination state-action pairs
    #            v_peraction.append(0)
    #        else:
    #            v_peraction.append(
    #                sum([p * (reward(s, a) + gamma * V[s1]) for s1, p in transition_prob[s][a]])
    #            )
    #    return max(v_peraction)


