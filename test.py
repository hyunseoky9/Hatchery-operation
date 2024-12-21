# run sarsa algorithms to come up with optimal policy
import pickle
from numba import jit
from td_learning import td_learning
from env1_0 import Env1_0
from scipy.stats import poisson, norm,binom
import random
import pandas as pd
import numpy as np
random.seed(23)
np.random.seed(23)

td = td_learning([-1,-1,-1,-1,-1,-1],2)
Q, policy, Q_update_counter = td.td_lambda(num_episodes=1,sarsaoption=1,lam=0.92)


    #def td_lambda(self, num_episodes,sarsaoption):
    #def td_n(self, num_episodes,sarsaoption, n):


