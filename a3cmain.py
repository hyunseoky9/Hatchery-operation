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
from A3C import *
from env1_0 import Env1_0
from rich.traceback import install
install()

if __name__ == "__main__":

    seednum = 432 #random.randint(0,1000)
    
    print(f'seed: {seednum}')
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)

    env = Env1_0([-1,-1,-1,-1,-1,-1],2,1)
    contaction = 0
    lr = 0.00002
    lrdecaytype = 'lin' # exp= exponential, lin= linear (original paper)
    lrdecayrate = 0.9992
    min_lr = 0
    tmax = 4
    Tmax = 1000000 #10**7
    lstm = 0 # lstm layer option 
    normalize = False
    calc_MSE = True
    calc_perf = True
    SavePolicyCycle = int(Tmax/20) # number of T steps to save policy network (global network).
    MSEV, MSEP, final_avgreward = A3C(env,contaction,lr,lrdecaytype,lrdecayrate,min_lr,normalize,calc_MSE,calc_perf,tmax,Tmax,lstm,SavePolicyCycle,seednum)