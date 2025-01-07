import shutil
import subprocess
import os
import torch
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

def A3C(env,contaction,lr,lrdecayrate,normalize,calc_MSE,external_testing,tmax,Tmax):
    """
    N-step Advantage Actor-Critic (A3C) algorithm
    """
    # device for pytorch neural network
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    # parameters
    ## NN parameters
    state_size = len(env.statespace_dim) # state space dimension
    if contaction:
        action_size = 2
    else:
        action_size = env.actionspace_dim[0]
    hidden_size = 20
    hidden_num = 3

    gamma = env.gamma # discount rate
    max_steps = 1000 # max steps per episode
    ## A3C parameters    
    num_workers = 1 # number of workers
    Tmax = 100000 # total number of global steps
    t_max = 5 # number of steps before updating the global network
    

    ## performance testing sample size
    performance_sampleN = 1000
    final_performance_sampleN = 1000

    ## normalization parameters
    if env.envID == 'Env1.0': # for discrete states
        state_max = (torch.tensor(env.statespace_dim, dtype=torch.float32) - 1).to(device)
        state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32).to(device)
    elif env.envID in ['Env1.1','Env1.2']: # for continuous states
        state_max = torch.tensor([env.states[key][1] for key in env.states.keys()], dtype=torch.float32).to(device)
        state_min = torch.tensor([env.states[key][0] for key in env.states.keys()], dtype=torch.float32).to(device)

    # initialization

    # start testing process
    testwd = './a3c results/intermediate training policy network'
    # delete all the previous network files in the intermediate network folder to not test the old Q networks
    for file in os.listdir(testwd):
        try:
            os.remove(os.path.join(testwd,file))
        except PermissionError:
            print(f"File {filepath} is locked. Retrying...")
            time.sleep(5)  # Wait 5 second
            os.remove(filepath)  # Retry deletion
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
        initlist = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
        reachable_states = torch.tensor([env._unflatten(i[0]) for i in reachables], dtype=torch.float32).to(device)
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64).to(device)
        reachable_actions = torch.tensor([i[1] for i in reachables], dtype=torch.int64).unsqueeze(1).to(device)
    elif env.envID == 'Env1.1':
        initlist = [-1,-1,-1,-1,-1,-1]

    ## initialize performance metrics
    # load Q function from the value iteration for calculating MSE
    if calc_MSE:
        if env.envID == 'Env1.0':
            with open(f"value iter results/Q_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
                Q_vi = pickle.load(file)
            Q_vi = torch.tensor(Q_vi[reachable_uniquestateid].flatten(), dtype=torch.float32).to(device)
    MSE = []
    # initialize reward performance receptacle
    avgperformances = [] # average of rewards over 100 episodes with policy following trained Q
    final_avgreward = 0
    print(f'performance sampling: {performance_sampleN}/{final_performance_sampleN}')

    mp.set_start_method("spawn", force=True)

    

def worker(global_net, optimizer, T, worker_id, envinit_params, networkinit_params, worker_params):
    """Worker proccess for A3C using Hogwild!"""
    # Initialize local environment and network
    env = Env1_0(**envinit_params)
    local_net = A3CNN(
        state_size = networkinit_params['state_size'],
        contaction = networkinit_params['contaction'],
        action_size = networkinit_params['action_size'],
        hidden_size = networkinit_params['hidden_size'],
        hidden_num = networkinit_params['hidden_num'],
        lr = networkinit_params['lr'],
        lrdecayrate = networkinit_params['lrdecayrate'],
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
    state = torch.tensor(env.state, dtype=torch.float32)
    episode_reward = 0

    while True:
        # Store transitions
        states, actions, rewards = [], [], []
        for _ in range(tmax):
            with torch.no_grad():
                policy, value, _ = local_net(state.unsqueeze(0))
            action = torch.multinomial(policy, 1).item() # sample action
            action_prob = policy[0, action]
            reward, done, _ = env.step(action)

            # save transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = torch.tensor(env.state, dtype=torch.float32)

            with T.get_lock(): # safely read and update global counter
                T.value += 1
                if T.value >= Tmax:
                    return
            
            episode_reward += reward
            if done:
                state = torch.tensor(env.reset([-1,-1,-1,-1,-1,-1]), dtype=torch.float32)
                print(f"Worker {worker_id} episode reward: {episode_reward}")
                episode_reward = 0
                break
        
        # Compute n-step returns and advantages
        R = 0 if done else local_net(state.unsqueeze(0))[-1].item() # Bootstrapped value
        returns, advantages = [], []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.stack([local_net(s.unsqueeze(0))[1] for s in states]).squeeze()
        advantages = returns - values

        # compute losses
        policy_loss, value_loss, entropy_loss = 0,0,0
        for s, a, adv, R in zip(states, actions, advantages, returns):
            policy, value, _ = local_net(s.unsqueeze(0))  # Use the correct state
            log_prob = torch.log(policy[0, a])  # Use the correct action
            entropy = -torch.sum(policy * torch.log(policy))  # Entropy of the policy
            policy_loss += -log_prob * adv
            value_loss += (value - R) ** 2
            entropy_loss += entropy
        
        total_loss = policy_loss + l * value_loss - beta * entropy_loss

        # back propagation
        optimizer.zero_grad()
        total_loss.backward()
        for global_param, local_param in zip(global_net.parameters(), local_net.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad # Apply gradients directly to global net
        optimizer.step()

        # Sync local network with global network
        local_net.load_state_dict(global_net.state_dict())

