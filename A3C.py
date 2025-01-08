import shutil
import subprocess
import os
import torch
from IPython.display import display
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR
import torch.multiprocessing as mp
import numpy as np
import random
from calc_performance import *
import pickle
from choose_action import *
from A3CNN import *
from env1_0 import Env1_0
from env1_1 import Env1_1
import time

def A3C(env,contaction,lr,min_lr,normalize,calc_MSE,external_testing,tmax,Tmax,lstm):
    """
    N-step Advantage Actor-Critic (A3C) algorithm
    """
    # parameters
    ## NN parameters
    state_size = len(env.statespace_dim) # state space dimension
    if contaction:
        action_size = 2
    else:
        action_size = env.actionspace_dim[0]
    hidden_size = 20
    hidden_num = 3
    ## LSTM parameters
    lstm_num = 20

    gamma = env.gamma # discount rate
    max_steps = 1000 # max steps per episode
    ## A3C parameters    
    num_workers = 2 # number of workers
    #tmax = 5 # number of steps before updating the global network
    l = 0.5 # weight for value loss
    beta  = 0.01 # weight for entropy loss

    ## performance testing sample size
    performance_sampleN = 1000
    final_performance_sampleN = 1000

    ## normalization parameters
    if env.envID == 'Env1.0': # for discrete states
        state_max = (torch.tensor(env.statespace_dim, dtype=torch.float32) - 1)
        state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32)
    elif env.envID in ['Env1.1','Env1.2']: # for continuous states
        state_max = torch.tensor([env.states[key][1] for key in env.states.keys()], dtype=torch.float32)
        state_min = torch.tensor([env.states[key][0] for key in env.states.keys()], dtype=torch.float32)

    # initialization
    ## start testing process
    testwd = './a3c results/intermediate training policy network'
    # delete all the previous network files in the intermediate network folder to not test the old Q networks
    for file in os.listdir(testwd):
        try:
            os.remove(os.path.join(testwd,file))
        except PermissionError:
            print(f"File {testwd} is locked. Retrying...")
            time.sleep(5)  # Wait 5 second
            os.remove(testwd)  # Retry deletion
    # run testing script in a separate process if external testing is on
    if external_testing:
        # run the testing script in a separate process
        # Define the script and arguments
        script_name = "performance_tester.py"
        args = ["--num_episodes", f"{0}", "--DQNorPolicy", "1", "--envID", f"{env.envID}",
                 "--parset", f"{env.parset+1}", "--discset", f"{env.discset}", "--midsample", f"{performance_sampleN}",
                 "--finalsample", f"{final_performance_sampleN}","--initQperformance", f"{0}"]
        # Run the script independently with arguments
        #subprocess.Popen(["python", script_name] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["python", script_name] + args)
    
    ## state initialization setting 
    if env.envID == 'Env1.0':
        initstate = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
        reachable_states = torch.tensor([env._unflatten(i[0]) for i in reachables], dtype=torch.float32)
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64)
        reachable_actions = torch.tensor([i[1] for i in reachables], dtype=torch.int64).unsqueeze(1)
    elif env.envID == 'Env1.1':
        initstate = [-1,-1,-1,-1,-1,-1]

    ## initialize performance metrics
    # load Q function from the value iteration for calculating MSE
    if calc_MSE:
        if env.envID == 'Env1.0':
            with open(f"value iter results/V_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
                V_vi = pickle.load(file)
            V_vi = torch.tensor(V_vi[reachable_uniquestateid].flatten(), dtype=torch.float32)
            with open(f"value iter results/policy_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
                policy_vi = pickle.load(file)
            policy_vi = torch.tensor(policy_vi[reachable_uniquestateid].flatten(), dtype=torch.float32)
    MSEV = [] # MSE for value function
    MSEP = [] # MSE for policy function
    # initialize reward performance receptacle
    avgperformances = [] # average of rewards over 100 episodes with policy following trained Q
    print(f'performance sampling: {performance_sampleN}/{final_performance_sampleN}')

    # Set multiprocessing method
    mp.set_start_method("spawn", force=True)

    # Initialize shared global network and optimizer
    global_net = A3CNN(state_size, contaction, action_size, hidden_size, hidden_num, lstm, lstm_num, normalize, state_min, state_max)
    global_net.share_memory() # Share network across processes
    optimizer = torch.optim.Adam(global_net.parameters(), lr=lr)

    # Global counter 
    T = mp.Value('i', 0)

    # Environment parameters
    envinit_params = {
        "envID": env.envID,
        "initstate": [-1,-1,-1,-1,-1,-1],
        "parameterization_set": env.parset + 1,
        "discretization_set": env.discset
    }
    # networkinit_params
    networkinit_params = {
        "state_size": state_size,
        "contaction": 0,
        "action_size": action_size,
        "hidden_size": hidden_size,
        "hidden_num": hidden_num,
        "lstm": 0,
        "lstm_num": 0,
        "normalize": normalize,
        "state_min": state_min,
        "state_max": state_max
    }
    # worker_params
    worker_params = {
        "tmax": tmax,
        "Tmax": Tmax,
        "gamma": gamma,
        "l": l,
        "beta": beta,
        "lr": lr,
        "min_lr": min_lr,
        "max_steps": max_steps
    }

    # Spawn worker processes
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(global_net, optimizer, T, worker_id, envinit_params, networkinit_params, worker_params))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print('workers done')
    # make outputs
    ## calculate final average reward
    ## calculate MSE
    ## Save the final network
    ## make discrete Q if the env is discrete and save
    Q_discrete = None
    return MSE, final_avgreward

def adjust_learning_rate(optimizer, T, Tmax, initial_lr, min_lr):
    """Adjust learning rate based on the global step."""
    lr = initial_lr * (1 - T.value / Tmax)
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, min_lr)

def worker(global_net, optimizer, T, worker_id, envinit_params, networkinit_params, worker_params):
    """Worker proccess for A3C using Hogwild!"""
    print(f"Worker {worker_id} initiated")
    # Initialize local environment and network
    initstate = envinit_params['initstate']
    if envinit_params['envID'] == 'Env1.0':
        env = Env1_0(envinit_params['initstate'], envinit_params['parameterization_set'], envinit_params['discretization_set'])
    elif envinit_params['envID'] == 'Env1.1':
        env = Env1_1(envinit_params['initstate'], envinit_params['parameterization_set'], envinit_params['discretization_set'])

    local_net = A3CNN(
        state_size = networkinit_params['state_size'],
        contaction = networkinit_params['contaction'],
        action_size = networkinit_params['action_size'],
        hidden_size = networkinit_params['hidden_size'],
        hidden_num = networkinit_params['hidden_num'],
        lstm = networkinit_params['lstm'],
        lstm_num = networkinit_params['lstm_num'],
        normalize = networkinit_params['normalize'],
        state_min = networkinit_params['state_min'],
        state_max = networkinit_params['state_max'],
    )
    local_net.load_state_dict(global_net.state_dict()) # copy global network weights

    # Initialize local environment and network
    tmax = worker_params['tmax']
    Tmax = worker_params['Tmax']
    gamma = worker_params['gamma']
    l = worker_params['l']
    beta = worker_params['beta']
    lr = worker_params['lr']
    min_lr = worker_params['min_lr']

    episode_reward = 0

    # Initialize LSTM hidden state
    hidden_state = None
    done = True
    episode_count = 0
    while True:
        # Store transitions and hidden states
        states, actions, rewards = [], [], []
        hidden_in = []
        # If episode is done, reset environment & hidden states
        if done:
            env.reset(initstate)
            state = torch.tensor(env.state, dtype=torch.float32)
            hidden_state = None
            print(f"Worker {worker_id} episode {episode_count} reward: {episode_reward}")
            episode_count += 1
            episode_reward = 0

        for _ in range(tmax):
            # Store the hidden state *before* the forward pass
            hidden_in.append(hidden_state)
            with torch.no_grad():
                policy, value, hidden_state = local_net(state.unsqueeze(0), hidden_state)

            action = torch.multinomial(policy, 1).item() # sample action
            
            reward, done, _ = env.step(action)

            # save transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = torch.tensor(env.state, dtype=torch.float32)

            with T.get_lock(): # safely read and update global counter
                T.value += 1
                adjust_learning_rate(optimizer, T, Tmax, lr, min_lr)  # Adjust learning rate dynamically
                if T.value >= Tmax:
                    return
            
            episode_reward += reward
            if done:
                break
        
        # Compute n-step returns and advantages
        if done:
            R = 0
        else:
            with torch.no_grad():
                _, value_next, _ = local_net(state.unsqueeze(0), hidden_state)
            R = value_next.item()
        
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # compute losses
        optimizer.zero_grad()
        policy_loss, value_loss, entropy_loss = 0.0 ,0.0 ,0.0
        hidden_state_train = hidden_in[0]
        for s, a, R in zip(states, actions, returns):
            policy, value, hidden_state_train = local_net(s.unsqueeze(0), hidden_state_train)  # Use the correct state
            log_prob = torch.log(policy[0, a] + 1e-13)  # Use the correct action
            entropy = -torch.sum(policy * torch.log(policy + 1e-13), dim=1)  # Entropy of the policy
            advantage = R - value
            policy_loss += -log_prob * advantage
            value_loss += advantage.pow(2)
            entropy_loss += entropy.mean()
        
        total_loss = policy_loss + l * value_loss - beta * entropy_loss

        # back propagation
        total_loss.backward()
        for global_param, local_param in zip(global_net.parameters(), local_net.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad # Apply gradients directly to global net
        optimizer.step()

        # Sync local network with global network
        local_net.load_state_dict(global_net.state_dict())

def tester(MSEV, MSEP, avgperformances, V_vi, policy_vi, T, global_net, envinit_params, networkinit_params, calc_MSE, performance_sampleN, final_performance_sampleN):
    """
    Test the performance of the policy network in 3 ways:
    1. Calculate the MSE of the value function
    2. Calculate the MSE of the policy function
    3. Calculate the average reward over N episodes (performance_sampleN, final_performance_sampleN)
    """
    if calc_MSE:
    
        
