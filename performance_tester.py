# compute performance of the Q networks that are output during the training.
import argparse
import time
from env1_0 import Env1_0
from calc_performance import *
import torch
from torch import nn
import numpy as np
import sys
import os
import random

seednum = 12 #random.randint(0,1000) # 12
random.seed(seednum)
np.random.seed(seednum)
torch.manual_seed(seednum)

# Define the arguments
parser = argparse.ArgumentParser(description="Example script.")
parser.add_argument("--num_episodes", type=str, required=True, help="Argument 1")
parser.add_argument("--DQNorPolicy", type=str, required=True, help="Argument 2")
args = parser.parse_args()

num_episode = int(args.num_episodes)
DQNorPolicy = int(args.DQNorPolicy)
#num_episode = int(sys.argv[1])
#DQNorPolicy = int(sys.argv[2]) # 0 for DQN, 1 for Policy gradient
envID = 1.1
interval = 1000
if envID == 1.1:
    env = Env1_0([-1,-1,-1,-1,-1,-1],2,1)
elif envID == 1.1:
    foo = 0
print(f'num_episode: {num_episode} DQNorPolicy: {DQNorPolicy} env: {envID}')
avgperformances = []
if DQNorPolicy == 0:
    wd = './deepQN results/training Q network'
        
    for i in range(0,num_episode+1,interval):
        print(f'episode {i}')
        filename= f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN_episode{i}.pt"
        print(filename)
        filefound = 0
        # if file is not found, the algorithm is still running, wait and try again
        while filefound == 0:
            try:
                Q = torch.load(filename,weights_only=False)
                filefound = 1
                print('Q loaded successfully')
            except:
                print('file not found, sleeping 3 sec')
                time.sleep(3)
        # calculate performance 
        if i == num_episode: # Final Q network performance sampled more accurately.
            print('calculating final performance')
            performance = calc_performance(env,Q=Q,episodenum=10000)
        else:
            print('calculating performance')
            performance = calc_performance(env,Q=Q,episodenum=1000)
            print('finished calculating performance')
        avgperformances.append(performance)
else: # fill this in later when you have policy gradient algorithms!
    foo = 0 

# save the performance results
if DQNorPolicy == 0:
    if env.envID == 'Env1.0':
        wd2 = './deepQN results'
        np.save(f"{wd2}/rewards_{env.envID}_par{env.parset}_dis{env.discset}_DQN.npy", avgperformances)
else:
    foo = 0
