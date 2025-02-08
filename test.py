import numpy as np
import torch
from env2_1 import Env2_1
from choose_action import choose_action
from choose_action_a3c import choose_action_a3c
from calc_performance import calc_performance
optim_method = [0, 66102, 36]
env = Env2_1([-1,-1,-1,-1,-1,-1],2,0)
wd = f'./DRQN results/seed{optim_method[1]}_paramset{optim_method[2]}'
method_str = 'DRQN'
filename= f"{wd}/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN.pt"
# if you want to test with the best Q network, comment out the next two lines
episodenum = 2900
filename= f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN_episode{episodenum}.pt"
Q = torch.load(filename,weights_only=False)

device=  'cpu'
print(calc_performance(env, device, Q, policy=None, episodenum=1000, t_maxstep = 100, drqn=True, actioninput=False))