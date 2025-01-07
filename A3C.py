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
    # DQN
    state_size = len(env.statespace_dim) # state space dimension
    if contaction:
        action_size = 2
    else:
        action_size = env.actionspace_dim[0]
    hidden_size = 20
    hidden_num = 3

    gamma = env.gamma # discount rate
    max_steps = 1000 # max steps per episode
    
    num_workers = 1 # number of workers

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
        args = ["--num_episodes", f"{num_episodes}", "--DQNorPolicy", "1", "--envID", f"{env.envID}",
                 "--parset", f"{env.parset+1}", "--discset", f"{env.discset}", "--midsample", f"{performance_sampleN}",
                 "--finalsample", f"{final_performance_sampleN}","--initQperformance", f"{initperform}"]
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
