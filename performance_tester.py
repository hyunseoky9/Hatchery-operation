# compute performance of the Q networks that are output during the training.
import shutil
import argparse
import time
from env1_0 import Env1_0
from env1_1 import Env1_1
from env2_0 import Env2_0
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

# device for pytorch neural network
device = (
"cuda"
if torch.cuda.is_available()
else "mps"
if torch.backends.mps.is_available()
else "cpu"
)
print(f"Using {device} device")

# Define the arguments
parser = argparse.ArgumentParser(description="Example script.")
parser.add_argument("--num_episodes", type=str, required=True, help="Argument 1")
parser.add_argument("--DQNorPolicy", type=str, required=True, help="Argument 2")
parser.add_argument("--envID", type=str, required=True, help="Argument 3")
parser.add_argument("--parset", type=str, required=True, help="Argument 4")
parser.add_argument("--discset", type=str, required=True, help="Argument 5")
parser.add_argument("--midsample", type=str, required=True, help="Argument 6")
parser.add_argument("--finalsample", type=str, required=True, help="Argument 7")
parser.add_argument("--initQperformance", type=str, required=True, help="Argument 8")
parser.add_argument("--maxstep", type=str, required=True, help="Argument 9")
parser.add_argument("--drqn", type=str, required=True, help="Argument 10")
parser.add_argument("--actioninput", type=str, required=True, help="Argument 11")

args = parser.parse_args()

num_episode = int(args.num_episodes)
DQNorPolicy = int(args.DQNorPolicy)
envID = args.envID
parset = int(args.parset)
discset = int(args.discset)
midsample = int(args.midsample)
finalsample = int(args.finalsample)
initQperformance = float(args.initQperformance)
maxstep = int(args.maxstep)
drqn = int(args.drqn)
actioninput = int(args.actioninput)
if drqn == 1:
    drqn = True
else:
    drqn = False
if actioninput == 1:
    actioninput = True
else:
    actioninput = False
print(f'num_episode: {num_episode} DQNorPolicy: {DQNorPolicy} env: {envID} parset: {parset} discset: {discset}')
print(f'midsample size: {args.midsample} finalsample size: {args.finalsample}')
#num_episode = int(sys.argv[1])
#DQNorPolicy = int(sys.argv[2]) # 0 for DQN, 1 for Policy gradient
interval = 1000
if envID == 'Env1.0':
    env = Env1_0([-1,-1,-1,-1,-1,-1],parset,discset)
elif envID == 'Env1.1':
    env = Env1_1([-1,-1,-1,-1,-1,-1],parset,discset)
elif envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','Env2.7','tiger']:
    env = Env2_0([-1,-1,-1,-1,-1,-1],parset,discset)
avgperformances = []
if DQNorPolicy == 0:
    if drqn == False:
        wd = './deepQN results/intermediate training Q network'
        method_str = 'DQN'
    else:
        wd = './DRQN results/intermediate training Q network'
        method_str = 'DRQN'

    for i in range(0,num_episode+1,interval):
        print(f'episode {i}')
        filename= f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_{method_str}_episode{i}.pt"
        print(filename)
        filefound = 0
        # if file is not found, the algorithm is still running, wait and try again
        trynum = 0
        while filefound == 0:
            try:
                Q = torch.load(filename,weights_only=False)
                filefound = 1
                print('Q loaded successfully')
            except:
                print('file not found, sleeping 3 sec')
                time.sleep(3)
                trynum += 1
            if trynum > 30:
                print('file not found after 30 tries, exiting')
                sys.exit()
        # calculate performance 
        if i == num_episode: # Final Q network performance sampled more accurately.
            print('calculating final performance')
            performance = calc_performance(env,device,Q=Q,episodenum=midsample,t_maxstep=maxstep,drqn=drqn,actioninput=actioninput)
        else:
            print('calculating performance')
            performance = calc_performance(env,device,Q=Q,episodenum=finalsample,t_maxstep=maxstep,drqn=drqn,actioninput=actioninput)
            print('finished calculating performance')
        avgperformances.append(performance)
else: # fill this in later when you have policy gradient algorithms!
    foo = 0 

# save the performance results and the best Q network
if DQNorPolicy == 0: # DQN
    if drqn == False:
        wd2 = './deepQN results'
    else:
        wd2 = './DRQN results'
    np.save(f"{wd2}/rewards_{env.envID}_par{env.parset}_dis{env.discset}_{method_str}.npy", avgperformances)
    # best model
    if initQperformance < max(avgperformances):
        bestidx = np.array(avgperformances).argmax()
        bestfilename = f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_{method_str}_episode{bestidx*1000}.pt"
        shutil.copy(bestfilename, f"{wd2}/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_{method_str}.pt")
else: # policy gradient
    foo = 0
