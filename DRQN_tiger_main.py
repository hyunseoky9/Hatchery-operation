import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR
import os
from IPython.display import display
import pickle
import math
import numpy as np
import random
from  DRQN import *
from tiger import Tiger
import pandas as pd
paramid = 24
# process hyperparameter dataframe
hyperparameterization_set_filename = './hyperparamsets/DRQN_hyperparameters.csv'
paramdf = pd.read_csv(hyperparameterization_set_filename, header=None)
paramdf = paramdf.T
paramdf.columns = paramdf.iloc[0]
paramdf = paramdf.drop(0)
if paramdf['seed'].iloc[paramid] == 'random':
    seednum = random.randint(0,100000)
else:
    seednum = int(paramdf['seed'].iloc[paramid])

print(f'seed: {seednum}')
random.seed(seednum)
np.random.seed(seednum)
torch.manual_seed(seednum)

env = Tiger([-1])
# basic settings
num_episodes = int(paramdf['episodenum'].iloc[paramid])

epdecayopt = int(paramdf['epsilon'].iloc[paramid].split(';')[0])
lr = float(paramdf['lr'].iloc[paramid])
lrdecayrate = float(paramdf['lrdecay'].iloc[paramid])
if paramdf['minlr'].iloc[paramid] == 'inf':
    minlr = float('-inf')
normalize = bool(int(paramdf['normalize'].iloc[paramid]))
## DQN extra extensions
DDQN = bool(int(paramdf['ddqn'].iloc[paramid]))
nstep = int(paramdf['nstep'].iloc[paramid])
distributional = bool(int(paramdf['distributional'].iloc[paramid]))

# training cycles
training_cycle = int(paramdf['training_cycle'].iloc[paramid])
target_update_cycle = int(paramdf['target_update_cycle'].iloc[paramid])
# performance evaluation
external_testing = bool(int(paramdf['external testing'].iloc[paramid]))
# Q initialization
bestQinit = bool(int(paramdf['bestQinit'].iloc[paramid]))
# option to input action in the network.
actioninput = bool(int(paramdf['actioninput'].iloc[paramid]))
# option to always sample sequences from the start of an episode
samplefromstart = bool(int(paramdf['samplefromstart'].iloc[paramid]))

print('successfully ran the code')
#rewards, final_avgreward = DRQN(env,num_episodes,epdecayopt,
#    DDQN,nstep,distributional,lrdecayrate,lr,minlr,training_cycle,target_update_cycle, external_testing,normalize,bestQinit,actioninput,samplefromstart, paramdf, paramid)
