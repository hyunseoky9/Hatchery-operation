from numba import jit
import multiprocessing
import logging
import itertools
import pickle
from env1_0 import Env1_0
import numpy as np
from scipy.stats import norm, beta, binom, poisson
import scipy.stats as stats
from IPython.display import display
from math import floor

class td_learning(Env1_0):
    # optimizing policy using temporal difference learning
    # options: Sarsa(lambda), Q-Learning(lambda), n-step sarsa, n-step Q-learning
    def td_lambda(self,num_episodes,sarsaoption,lam):
        # train the agent using Q-Learning(lambda)
        # sarsaoption = 0 for Sarsa(lambda) and 1 for Q-Learning(lambda) 
        # num_episodes = number of episodes to train the agent
        lr = 0.1 # learning rate
        Q = np.zeros([np.prod(self.statespace_dim), len(self.actions["a"])]) # initialize Q function
        Q_update_counter = Q.copy()
        theta = 0.00001
        i = 0
        delta = np.inf
        while i < num_episodes: #delta > theta:
            ep = 1/(i+1) #self.epsilon_update(i) # epsilon
            self.state = self.reset([-1,-1,-1,-1,-1,-1])
            S = self._flatten(self.state)
            done = False
            e = np.zeros([np.prod(self.statespace_dim), len(self.actions["a"])])
            #Q_old = Q.copy()
            t = 1
            while (done == False):
                # take action based on epsilon greedy method.
                a = self.choose_action(S, Q, ep)
                reward, done, survival_rate = self.step(a)
                S_next = self._flatten(self.state)
                # Calculate temporal-difference error
                if(sarsaoption == 0): # sarsa
                    a_next = self.choose_action(S, Q, ep)
                    delta = reward + self.gamma*Q[S_next,a_next] - Q[S,a]
                else: # Q-Learning
                    delta = reward + self.gamma*np.max(Q[S_next,:]) - Q[S,a]
                
                # update eligibility trace
                e[S,a] = e[S,a] + 1
                # update Q function for each S state
                Q = Q + lr*delta*e
                #Q_update_counter = Q_update_counter + e
                # see if Q values where update counter is 0 is not 0. 

                # decay eligibility trace
                e = e*self.gamma*lam
                # move to next step
                S = S_next
                self.state = self._unflatten(S_next)
                t += 1
            if i % 1000 == 0:
                display(f"episode: {i}")
            i += 1
        policy = self._get_policy(Q)
        
        # save Q and policy
        if sarsaoption == 0:
            sarsastr = f"sarsa(lambda{lam})"
        else:
            sarsastr = f"Qlearning(lambda{lam})"
        
        wd = './td results'
        with open(f"{wd}/Q_{self.envID}_par{self.parset}_{sarsastr}.pkl", "wb") as file:
            pickle.dump(Q, file)
        with open(f"{wd}/policy_{self.envID}_par{self.parset}_{sarsastr}.pkl", "wb") as file:
            pickle.dump(policy, file)
        return Q, policy, Q_update_counter



    def td_n(self, sarsaoption, n):
        # train the agent using Sarsa(n) or Sarsamax(n)
        # sarsaoption = 0 for Sarsa and 1 for Q-Learning 
        # num_episodes = number of episodes to train the agent
        # n = number of steps to look ahead
        lr = 0.01
        Q = np.zeros([np.prod(self.statespace_dim), len(self.actions["a"])]) # initialize Q function
        theta = 0.01
        i = 0
        delta = np.inf
        while delta > theta:
            ep = 1/(i+1) # epsilon
            self.state = self.reset([-1,-1,-1,-1,-1,-1])
            S0 = self._flatten(self.state)
            done = 0
            T = np.inf
            t = 0
            a0 = self.choose_action(S, Q, ep)
            a_store = [a0]
            s_store = [S0] 
            r_store = [0]
            Q_old = Q.copy()
            t2done = 0
            while t2done == 0:
                if t < T:
                    # take action based on epsilon greedy method.
                    reward, done, survival_rate = self.step(a)
                    S_next = self._flatten(self.state)
                    a_store.append(a)
                    s_store.append(S_next)
                    r_store.append(reward)
                    if done:
                        T = t + 1
                    else: 
                        a = self.choose_action(S_next, Q, ep)
                t2 = t - self.n + 1 # tau here is not the tau in the model
                if t2 >= 0:
                    G = np.sum([self.gamma**(i-t2-1)*self.rewards[i] for i in np.arange(t2+1, min(t2+self.n, T)+1)])
                    if t2 + self.n < T:
                        if sarsaoption == 0:
                            G = G + self.gamma**n * Q[s_store[t2+n], a_store[t2+n]]
                        else: 
                            G = G + self.gamma**n * np.max(Q[s_store[t2+n],:])
                    Q[s_store[t2], a_store[t2]] = Q[s_store[t2], a_store[t2]] + lr*(G - Q[s_store[t2], a_store[t2]])
                if t2 == T - 1:
                    t2done = 1
            delta = np.max(np.abs(Q_old - Q))
            display(f"delta: {delta}")
            i += 1
        # derive policy from Q
        policy = self._get_policy(Q)

        # save Q and policy
        if sarsaoption == 0:
            sarsastr = "sarsa(n)"
        else:
            sarsastr = "Qlearning(n)"
        
        wd = './td results'
        with open(f"{wd}/Q_{self.envID}_par{self.parset}_{sarsastr}.pkl", "wb") as file:
            pickle.dump(Q, file)
        with open(f"{wd}/policy_{self.envID}_par{self.parset}_{sarsastr}.pkl", "wb") as file:
            pickle.dump(policy, file)

        return Q, policy


    def _get_policy(self,Q):
        # get policy from Q function
        policy = np.zeros(np.prod(self.statespace_dim))
        for i in range(np.prod(self.statespace_dim)):
            policy[i] = np.argmax(Q[i,:])
        return policy

    def choose_action(self, S, Q, ep):
        # take action based on epsilon greedy method.
        draw = np.random.binomial(1, ep)
        if draw == 0: # action based on policy
            return np.argmax(Q[S,:])
        else: # random action
            return np.random.choice(np.arange(0,len(self.actions["a"])))
 
    def epsilon_update(self, i):
        # update epsilon
        if i < 1000:
            return max(1/(i+1), 0.05)
        else:
            return 1/(i+1) 