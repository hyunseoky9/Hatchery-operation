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

def A3C(env,contaction,lr,lrdecaytype,lrdecayrate,min_lr,normalize,calc_MSE,calc_perf,tmax,Tmax,lstm,SavePolicyCycle,seednum):
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
    hidden_size = 10
    hidden_num = 2
    ## LSTM parameters
    lstm_num = 2

    gamma = env.gamma # discount rate
    max_steps = 1000 # max steps per episode
    ## A3C parameters    
    num_workers = 10 # number of workers
    #tmax = 5 # number of steps before updating the global network
    l = 1 # weight for value loss
    beta  = 0.1 # weight for entropy loss

    ## performance testing sample size
    performance_sampleN = 1000
    final_performance_sampleN = 1000

    # parameter print out
    print(f'env: {env.envID}  parset: {env.parset}  discset: {env.discset}')
    print(f'lr: {lr}  min_lr: {min_lr}  ')
    print(f'normalize: {normalize}')
    print(f'tmax: {tmax}  Tmax: {Tmax}')
    print(f'l: {l}  beta: {beta}')
    print(f'hiddensize: {hidden_size}  hiddennum: {hidden_num}')
    print(f'lstm: {lstm}')
    if lstm:
        print(f'lstm_num: {lstm_num}')
    print(f'num_workers: {num_workers}')
    print(f'calc_MSE: {calc_MSE}  calc_perf: {calc_perf}')
    ## normalization parameters
    if env.envID == 'Env1.0': # for discrete states
        state_max = (torch.tensor(env.statespace_dim, dtype=torch.float32) - 1)
        state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32)
    elif env.envID in ['Env1.1','Env1.2']: # for continuous states
        state_max = torch.tensor([env.states[key][1] for key in env.states.keys()], dtype=torch.float32)
        state_min = torch.tensor([env.states[key][0] for key in env.states.keys()], dtype=torch.float32)

    # initialization
    ## start testing process
    # output wd's
    testwd = './a3c results/intermediate training policy network' # for saving intermediate policy networks
    wd = './a3c results' # for saving all the main results
    # delete all the previous network files in the intermediate network folder to not test the old Q networks
    for file in os.listdir(testwd):
        try:
            os.remove(os.path.join(testwd,file))
        except PermissionError:
            print(f"File {testwd} is locked. Retrying...")
            time.sleep(5)  # Wait 5 second
            os.remove(testwd)  # Retry deletion
    
    ## state initialization setting 
    if env.envID == 'Env1.0':
        initstate = [-1,-1,-1,-1,-1,-1] # all random
        reachable_states = torch.tensor([env._unflatten(i) for i in env.reachable_states()], dtype=torch.float32) # unique reachable states in unflattened form. *Different from DQN!*
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64)
    elif env.envID == 'Env1.1':
        initstate = [-1,-1,-1,-1,-1,-1]

    ## initialize performance metrics
    # load Q function from the value iteration for calculating MSE
    if calc_MSE:
        if env.envID == 'Env1.0':
            with open(f"value iter results/V_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
                V_vi = pickle.load(file)
            V_vi = torch.tensor(V_vi[reachable_uniquestateid], dtype=torch.float32)
            with open(f"value iter results/policy_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
                policy_vi = pickle.load(file)
            policy_vi = torch.tensor(policy_vi[reachable_uniquestateid].flatten(), dtype=torch.float32)
    else:
        V_vi = None
        policy_vi = None
    numtests = len(list(range(0, Tmax+1, SavePolicyCycle)))
    MSEV = torch.zeros(numtests,dtype=torch.float32).share_memory_() # MSE for value function
    MSEP = torch.zeros(numtests,dtype=torch.float32).share_memory_() # MSE for policy function
    # initialize reward performance receptacle
    avgperformances = torch.zeros(numtests,dtype=torch.float32).share_memory_() # average of rewards over 100 episodes with policy following trained Q
    print(f'performance sampling: {performance_sampleN}/{final_performance_sampleN}')
    print('---------------------------------------------')
    # Set multiprocessing method
    mp.set_start_method("spawn", force=True)

    # Initialize shared global network and optimizer
    global_net = A3CNN(state_size, contaction, action_size, hidden_size, hidden_num, lstm, lstm_num, normalize, state_min, state_max)
    # save the initial policy in intermediate results
    torch.save(global_net, f"{testwd}/PolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_A3C_T0.pt")
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
        "contaction": contaction,
        "action_size": action_size,
        "hidden_size": hidden_size,
        "hidden_num": hidden_num,
        "lstm": lstm,
        "lstm_num": lstm_num,
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
        "lrdecaytype": lrdecaytype,
        "lrdecayrate": lrdecayrate,
        "min_lr": min_lr,
        "max_steps": max_steps,
        "testwd": './a3c results/intermediate training policy network',
        "wd": './a3c results',
        "SavePolicyCycle": SavePolicyCycle,
        "base_seed": seednum,
        "num_workers": num_workers
    }

    # start timer
    start_time = time.time()
    processes = []
    # Spawn tester process
    p = mp.Process(target=tester, args=(MSEV, MSEP, avgperformances, V_vi, policy_vi, reachable_states, envinit_params, worker_params, calc_MSE, calc_perf, performance_sampleN, final_performance_sampleN,seednum))
    p.start()
    processes.append(p)
    # Spawn worker processes
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(global_net, optimizer, T, worker_id, envinit_params, networkinit_params, worker_params))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print('workers done')
    # save the global network.
    ## save the last model
    torch.save(global_net, f"{wd}/PolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_A3C.pt")
    ## save MSEV and MSEP
    if calc_MSE:
        np.save(f"{wd}/MSEV_{env.envID}_par{env.parset}_dis{env.discset}_A3C.npy", MSEV)
        np.save(f"{wd}/MSEP_{env.envID}_par{env.parset}_dis{env.discset}_A3C.npy", MSEP)
    # finish timer and print
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time elapsed: {int(minutes)} minutes and {int(seconds)} seconds")
    return MSEV, MSEP, avgperformances

def adjust_learning_rate(optimizer, T, Tmax, initial_lr, lrdecaytype, lrdecayrate, min_lr):
    """Adjust learning rate based on the global step."""
    if lrdecaytype == 'lin':
        lr = initial_lr * (1 - T.value / Tmax)
    elif lrdecaytype == 'exp':
        lr = initial_lr * lrdecayrate
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, min_lr)

def worker(global_net, optimizer, T, worker_id, envinit_params, networkinit_params, worker_params):
    """Worker proccess for A3C using Hogwild!"""
    print(f"Worker {worker_id} initiated")
    # Set random seed
    worker_seed = worker_params['base_seed'] + (worker_id + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
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
    lrdecaytype = worker_params['lrdecaytype']
    lrdecayrate = worker_params['lrdecayrate']
    min_lr = worker_params['min_lr']
    testwd = worker_params['testwd']
    SavePolicyCycle = worker_params['SavePolicyCycle']
    max_steps = worker_params['max_steps']
    episode_reward = 0

    # Initialize LSTM hidden state
    hidden_state = None
    done = True
    episode_count = 0
    t = 0
    while True:
        # Store transitions and hidden states
        states, actions, rewards = [], [], []
        hidden_in = []
        # If episode is done, reset environment & hidden states
        if done:
            env.reset(initstate)
            state = torch.tensor(env.state, dtype=torch.float32)
            hidden_state = None
            #if episode_count % 100 == 0:
            #    print(f"Worker {worker_id} episode {episode_count} (t={t}) reward: {episode_reward}")
            episode_count += 1
            episode_reward = 0
            t = 0

        for _ in range(tmax):
            # Store the hidden state *before* the forward pass
            hidden_in.append(hidden_state)
            with torch.no_grad():
                policy, value, hidden_state = local_net(state.unsqueeze(0), hidden_state)

            action = torch.multinomial(policy, 1).item() # sample action
            
            reward, done, _ = env.step(action)
            t += 1
            if t >= max_steps:
                done = True
            # save transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = torch.tensor(env.state, dtype=torch.float32)

            # read and update global counter
            with T.get_lock(): 
                T.value += 1 # update global counter
                adjust_learning_rate(optimizer, T, Tmax, lr, lrdecaytype, lrdecayrate, min_lr)  # Adjust learning rate dynamically
                if T.value % 10000 == 0:
                    print(f"Global step (T) = {T.value},   learning rate:{optimizer.param_groups[0]['lr']}")
                
                # save policy (global_net) every SavePolicyCycle
                if T.value % SavePolicyCycle == 0:
                    if env.envID in ['Env1.0', 'Env1.1']:
                        torch.save(global_net, f"{testwd}/PolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_A3C_T{T.value}.pt")
                if T.value >= Tmax:
                    print(f"Worker {worker_id} done. did {episode_count} episodes")
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
            policy_loss += -log_prob * advantage.detach()
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

def tester(MSEV, MSEP, avgperformances, V_vi, policy_vi, reachable_states, envinit_params, worker_params, calc_MSE, calc_perf, performance_sampleN, final_performance_sampleN,seednum):
    """
    Test the performance of the policy network in 3 ways:
    1. Calculate the MSE of the value function
    2. Calculate the MSE of the policy function
    3. Calculate the average reward over N episodes (performance_sampleN, final_performance_sampleN)
    4. test tesnum times in fixed interval
    """
    print(f"Tester initiated")
    # Set seed
    testerseed = worker_params['base_seed'] + worker_params['num_workers'] + 1
    random.seed(testerseed)
    np.random.seed(testerseed)
    torch.manual_seed(testerseed)
    # parameters
    Tmax = worker_params['Tmax']
    savecyc = worker_params['SavePolicyCycle']
    # calcualte intervals in global counter where the performance is tested once.
    intervals = list(range(0, Tmax+1, savecyc))
    interval_idx = 0

    # Initialize envd
    if envinit_params['envID'] == 'Env1.0':
        env = Env1_0(envinit_params['initstate'], envinit_params['parameterization_set'], envinit_params['discretization_set'])
    elif envinit_params['envID'] == 'Env1.1':
        env = Env1_1(envinit_params['initstate'], envinit_params['parameterization_set'], envinit_params['discretization_set'])
    action_size = env.actionspace_dim[0]

    # working directory to load networks from
    testwd = worker_params['testwd']
    wd = worker_params['wd']
    filenames = []
    # test if the model hasn't been tested while the counter is within the interval
    for interval in intervals:
        # load saved network
        filename = f"{testwd}/PolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_A3C_T{interval}.pt"
        filenames.append(filename)
        trynum = 0
        while trynum < 10:
            try:
                local_net = torch.load(filename, weights_only=False)
                break
            except:
                #print(f'file not found at {interval}/{Tmax}, sleeping 2 sec')
                time.sleep(2)
                trynum += 1
            if trynum == 10:
                #print(f'file not found at {interval}/{Tmax}, causing error')
                raise FileNotFoundError
        if calc_MSE: # Warning: this may only work for FF version.
            with torch.no_grad():
                policy, V, _ = local_net(reachable_states)
            # Calculate the MSE of the value function
            msevval = torch.mean((V.T.squeeze(0) - V_vi) ** 2).item()
            MSEV[interval_idx] = msevval
            # Calculate the MSE of the policy function (compare using the expected policy)
            action_idx_expanded = torch.tensor(np.arange(0,action_size),dtype=torch.float32).unsqueeze(0).expand(len(reachable_states),-1)
            msepval = torch.mean((torch.sum(policy*action_idx_expanded, dim=1) - policy_vi) ** 2).item()
            MSEP[interval_idx] = msepval
            mse_calc_str = f'MSEV: {msevval}     MSEP: {msepval}'
        else:
            mse_calc_str = ''
        if calc_perf:
            if interval_idx < len(intervals) - 1:
                # calculate the average reward over N episodes
                avgperformance = calc_performance(env,None,seednum,None,local_net,performance_sampleN,seednum)
            else:
                avgperformance = calc_performance(env,None,seednum,None,local_net,final_performance_sampleN)
            performance_str = f"Avg Performance: {avgperformance}"
            avgperformances[interval_idx] = avgperformance
        else:
            performance_str = ''
        print(f"Testing.. {interval}/{Tmax} {mse_calc_str}     {performance_str}")
        # confirm testing at the interval
        interval_idx += 1
    
    # save the model with the best performance
    if calc_perf:
        bestidx = np.array(avgperformances).argmax()
        bestfilename = f"{testwd}/PolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_A3C_T{intervals[bestidx]}.pt"
        shutil.copy(bestfilename, f"{wd}/bestPolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_A3C.pt")

    


