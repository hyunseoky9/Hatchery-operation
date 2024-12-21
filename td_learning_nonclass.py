import os
from IPython.display import display
import pickle
import math
import numpy as np
from numba import jit
from numba import prange
import random
import pandas as pd

def td_lambda(env,num_episodes,sarsaoption,lam,Qinitopt,epdecayopt,lropt):
    # train the agent using Q-Learning(lambda)
    # sarsaoption = 0 for Sarsa(lambda) and 1 for Q-Learning(lambda) 
    # num_episodes = number of episodes to train the agent
    # lam = lambda value for eligibility trace if using Sarsa(lambda) or Q-Learning(lambda)
    # epdecayopt = 0 for inverse decay, 1 for inverse decay with a minimum of 0.05, 2 for pure exploration for 10% of the episodes
    # lropt = 0 for constant learning rate, 1 for inverse decay learning rate
    print('initializing Q')
    Q = init_Q(env,Qinitopt) # initialize Q function
    # algorithm performance measuring variables
    Q_update_counter = init_Q(env,0)
    Qchanges = []
    n_actions = len(env.actions["a"])
    totrewards = [] # total rewards for each episodes
    gamma = env.gamma
    i = 0
    delta = np.inf
    if env.envID == 'Env1.0':
        initlist = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
    elif env.envID == 'Env0.0':
        initlist = [-1,-1]
        reachables = []

    # load Q function from the value iteration if its available
    if env.envID == 'Env1.0':
        with open(f"value iter results/Q_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
            Q_vi = pickle.load(file)
    elif env.envID == 'Env0.0':
        with open(f"value iter results/Q_Env0.0_par0_valiter.pkl", "rb") as file:
            Q_vi = pickle.load(file)
    MSE = []



    # run through the episodes
    while i < num_episodes: #delta > theta:
        # set epsilon. 
        # 0 for inverse decay
        # 1 for inverse decay with a minimum of 0.05
        # pure exploration for 10% and then inverse decay
        lr = lr_update(i,num_episodes,lropt) # 0=constant, 1=inverse decay
        ep = epsilon_update(i,epdecayopt,num_episodes) 
        

        env.reset(initlist) # random initialization
        S = env._flatten(env.state)
        inita = np.random.randint(0, n_actions)

        done = False

        e = np.zeros([np.prod(env.statespace_dim), len(env.actions["a"])])
        t = 1
        #totreward = 0 # total reward in an episode
        # run through the steps in the episode
        if sarsaoption == 0: # sarsa
            a = choose_action(S, Q, ep, n_actions,Q_update_counter)

        while (done == False):
            # take action based on epsilon greedy method.
            if sarsaoption == 1: # Q-learning
                if t > 1:
                    a = choose_action(S, Q, ep, n_actions,Q_update_counter)
                else: # force-visit action at the start of an episode
                    a = inita
            reward, done, rate = env.step(a) # take a step
            #totreward += reward
            S_next = env._flatten(env.state) # turn 6 coordinate state into a single index (flatten)
            # Calculate temporal-difference error
            if(sarsaoption == 0): # sarsa
                if done == False:
                    a_next = choose_action(S, Q, ep, n_actions,Q_update_counter)
                    delta = reward + env.gamma*Q[S_next,a_next] - Q[S,a]
                    a = a_next
                else:
                    delta = reward - Q[S,a]
            else: # Q-Learning
                if done == False:
                    delta = reward + env.gamma*np.max(Q[S_next,:]) - Q[S,a]
                else:
                    delta = reward - Q[S,a]

            # update eligibility trace
            e[S,a] = e[S,a] + 1
            Q_update_counter[S,a] += 1.0

            # update Q function for each S state that's been visited in this episode
            # and decay eligibility trace
            Q, e = updateQne(Q,e,lr*delta,gamma*lam)

            # move to next state
            S = S_next
            env.state = env._unflatten(S_next)
            t += 1
        if i % 1000 == 0:
            MSE.append(calculate_MSE(env, Q,Q_vi))
        #totrewards.append(totreward)
        if i % 1000 == 0:
            display(f"episode: {i}")
        i += 1
    policy = _get_policy(env,Q)
    # save Q and policy
    if sarsaoption == 0:
        sarsastr = f"sarsa(lambda{lam})"
    else:
        sarsastr = f"Qlearning(lambda{lam})"
    
    wd = './td results'
    with open(f"{wd}/Q_{env.envID}_par{env.parset}_dis{env.discset}_{sarsastr}.pkl", "wb") as file:
        pickle.dump(Q, file)
    with open(f"{wd}/policy_{env.envID}_par{env.parset}_dis{env.discset}_{sarsastr}.pkl", "wb") as file:
        pickle.dump(policy, file)

    #log_file.close()
    return Q, policy, Q_update_counter, Qchanges, totrewards, MSE

@jit(nopython=True)
def updateQne(Q,e,lr_delta,gamma_lam):
    # update Q function for each S state
    change = lr_delta*e
    Q = Q + change
    e = e*gamma_lam
    return Q, e

@jit(nopython=True)
def update_counter(Q_update_counter,e):
    # update Q update counter
    Q_update_counter = Q_update_counter + e
    return Q_update_counter

def calculate_MSE(env,Q,Q_vi):
    if env.envID == 'Env1.0':
        reachables = env.reachable_state_actions()
        indices = np.array(reachables)
        diff = np.sum((Q_vi[indices[:, 0], indices[:, 1]] - Q[indices[:, 0], indices[:, 1]])**2)
        return 1/len(reachables)*diff
            
    elif env.envID == 'Env0.0':
        diff = np.sum((Q_vi - Q)**2)
        return 1/(np.prod(env.statespace_dim)*env.actionspace_dim[0])*diff

#@jit(nopython=True)
def choose_action(S, Q, ep, n_actions, Q_update_counter):
    # draw a random value from uniform(0,1)
    draw = np.random.rand()
    if draw < ep:
        # random action
        return np.random.randint(0, n_actions)
        # random action prioritizing less visited action
        #return sample_from_softmax(Q_update_counter[S,:],[])
    else:
        # policy-based action
        return np.argmax(Q[S, :])

def _get_policy(env,Q):
    # get policy from Q function
    policy = np.zeros(np.prod(env.statespace_dim))
    for i in range(np.prod(env.statespace_dim)):
        policy[i] = np.argmax(Q[i,:])
    return policy

def epsilon_update(i,option,num_episodes):
    # update epsilon
    if option == 0:
        # inverse decay
        return 1/(i+1)
    elif option == 1:
        # inverse decay with a minimum epsilon of 0.01
        return max(1/(i+1), 0.2)
    elif option == 2:
        # pure exploration for 10% of the episodes
        if i < num_episodes*0.1:
            return 1
        else:
            return max(1/(i-(np.ceil(num_episodes*0.1)-1)), 0.01)
    elif option == 3: # exponential decay
        a = 1/num_episodes*10
        return np.exp(-a*i)
    elif option == 4: # logistic decay
        fix = 100000
        a=0.1
        b=-10*1/fix*3
        c=-fix*0.4
        return max(a/(1+np.exp(-b*(i+c))), 0.01)

def lr_update(i,num_episodes,option):
    if option == 0: # constant learning rate
        return 0.4
    elif option == 1: # inverse decay learning rate (don't recommend) doesn't guarantee probability of 1 when i =0 (could be higher)
        return max(1/(i+10),0.01)
    elif option == 2: # exponential decay
        fix = 40000
        a = 0.1
        b = 1/fix*13
        return max(a*np.exp(-b*i),0.01)
    elif option == 3: # logistic decay
        fix = 100000
        a=0.4
        b=-10*1/fix*3
        c=-fix*0.4
        return max(a/(1+np.exp(-b*(i+c))), 0.01)
    elif option == 4: # logistic decay with a stepdown
        fix = 40000
        a=0.1
        b=-10*1/fix*3
        c=-fix*0.075
        if i < 60000:
            return max(a/(1+np.exp(-b*(i+c))), 0.01)
        else: 
            return max(a/(1+np.exp(-b*(i+c))), 0.001)

def init_Q(env,option):
    # initialize Q function
    if option == 0: # all zero Q
        return np.zeros([np.prod(env.statespace_dim), len(env.actions["a"])])
    elif option == 1: # initialize Q with immediate reward
        Q = np.zeros([np.prod(env.statespace_dim), len(env.actions["a"])])
        for i in range(np.prod(env.statespace_dim)):
            for j in range(len(env.actions["a"])):
                env.state = env._unflatten(i)
                reward, done, survival_rate = env.step(j)
                Q[i,j] = reward
        return Q
    elif option == 2: # initialize Q with immediate reward and the next expected reward.
        filename = f'initQopt2_{env.envID}_par{env.parset}_dis{env.discset}.pkl' # initialized Q with immediate reward and the next expected reward
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                Q = pickle.load(file)
        else:
            Q = np.zeros([np.prod(env.statespace_dim), len(env.actions["a"])])
            samplenum = 100
            gamma = env.gamma
            for i in range(np.prod(env.statespace_dim)):
                for j in range(len(env.actions["a"])):
                    start = env._unflatten(i)
                    env.state = start
                    reward, done, _ = env.step(j)
                    next_state = env.state
                    next_expreward = 0
                    if done == False:
                        for sample in range(samplenum):
                            maxreward = -100000000
                            for k in range(len(env.actions["a"])):
                                env.reset(next_state)
                                # sample next reward
                                next_reward, done, _ = env.step(k)
                                if done == True:
                                    next_expreward = next_reward    
                                    break
                                else:
                                    maxreward = max(next_reward,maxreward)
                            if done == True:
                                break
                            else:
                                next_expreward += maxreward
                        if done == False:
                            next_expreward = next_expreward/samplenum
                    Q[i,j] = reward + gamma*next_expreward
                if i % 100 == 0:
                    print(f'{i}/{np.prod(env.statespace_dim)}')
            # save the initialized Q
            with open(filename, "wb") as file:
                pickle.dump(Q, file)   
        return Q
    elif option == 3: # initialize Q with the exact Q values for the terminal states or the invalid actions and the rest with 0
        filename = f'initQopt3_{env.envID}_par{env.parset}_dis{env.discset}.pkl' # initialized Q with immediate reward and the next expected reward
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                Q = pickle.load(file)
        else:
            Q = np.zeros([np.prod(env.statespace_dim), len(env.actions["a"])])
            samplenum = 10
            gamma = env.gamma
            for i in range(np.prod(env.statespace_dim)):
                for j in range(len(env.actions["a"])):
                    env.state = env._unflatten(i)
                    if env.state[5] == 1 and env.state[2] < j:
                        invalid_action = True
                    else: 
                        invalid_action = False
                    reward, done, _ = env.step(j)
                    next_state = env.state
                    next_expreward = 0
                    if done == False:
                        if invalid_action == True:
                            next_expreward, done, _ = env.step(1)
                            Q[i,j] = reward + gamma*next_expreward
                        else: 
                            Q[i,j] = 0
                    else:
                        Q[i,j] = reward
                if i % 100 == 0:
                    print(f'{i}/{np.prod(env.statespace_dim)}')
            # save the initialized Q
            with open(filename, "wb") as file:
                pickle.dump(Q, file)   
        return Q
        
#@jit(nopython=True)
def softmax_prob(visit_count,reachable_states):
    if len(visit_count.shape) == 2:
        visit_count_reachable = visit_count[reachable_states, :]
        # Calculate the proportion visited
        total_visits = np.sum(visit_count_reachable)
        proportion_visited = visit_count_reachable / total_visits
        inverse_proportion = 1.0 - proportion_visited #.flatten()
        # Compute softmax probabilities
        tau = 0.1
        exp_values = np.exp(inverse_proportion / tau)
        probabilities_reachables = exp_values / np.sum(exp_values)
        # Initialize full probabilities array with zeros
        probabilities = np.zeros(visit_count.shape, dtype=np.float64)
        # Assign probabilities only to reachable states
        probabilities[reachable_states,:] = probabilities_reachables
        probabilities /= np.sum(probabilities)
        return probabilities.flatten()
    else:
        # Calculate the proportion visited
        total_visits = np.sum(visit_count)
        if total_visits == 0:
            return np.ones(visit_count.shape)/np.prod(visit_count.shape)
        proportion_visited = visit_count / total_visits
        inverse_proportion = 1.0 - proportion_visited #.flatten()
        # Compute softmax probabilities
        tau = 0.1
        exp_values = np.exp(inverse_proportion / tau)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities



def sample_from_softmax(visit_count,reachable_states):
    probabilities = softmax_prob(visit_count,reachable_states)
    flat_idx = np.random.choice(len(probabilities), p=probabilities)
    if len(visit_count.shape) == 2:
        state, action = np.unravel_index(flat_idx, visit_count.shape)
        return state, action
    else:
        return flat_idx
    