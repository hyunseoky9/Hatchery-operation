import numpy as np
import torch
from choose_action import choose_action
from choose_action_a3c import choose_action_a3c
def calc_performance(env, device, Q=None, policy=None, episodenum=1000, t_maxstep = 1000, drqn=False, actioninput=False):
    """
    calculate the performance of the agent in the environment.
    For DQN calculate performance with the Q network. (use Q variable in the function input)
    For Tabular Q learning and value iteration, calculate perofrmance using the policy table.
    For policy gradient methods, calculate performance using the policy network.
    """

    managed = 0 # for tiger environment
    surveyed = 0 # for tiger environment
    if env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6']:
        actiondist = np.zeros(env.actionspace_dim[0])
    avgrewards = 0
    action_size = env.actionspace_dim[0]
    if Q is not None:
        distributional = Q.distributional
    fstack = 1 if not hasattr(Q, 'fstack') else Q.fstack

    for i in range(episodenum):
        rewards = 0
        if env.envID in ['Env1.0','Env1.1']:
            env.reset([-1,-1,-1,-1,-1,-1])
            hx = None # for A3C + lstm
        elif env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','tiger']:
            env.reset([-1,-1,-1,-1,-1,-1])
            hx = None # for A3C + lstm and RDQN

        if env.partial == False:
            stack = env.env*fstack
        else:
            stack = env.obs*fstack
        
        previous_action = 0
        done = False
        t = 0
        while done == False:
            if env.partial == False:
                state = env.state
            else:
                state = env.obs
            stack = stack[len(env.state):] + env.state

            if Q is not None:
                prev_a = previous_action if actioninput else None
                if drqn == True: #DRQN
                    action, hx = choose_action(state,Q,0,action_size,distributional,device, drqn, hx, prev_a)
                    if env.envID == 'tiger':
                        if action == 1:
                            managed = 1
                        if action == 2:
                            surveyed = 1
                    elif env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6']:
                        actiondist[action] += 1
                else: # DQN
                    action = choose_action(stack,Q,0,action_size,distributional,device, drqn, hx, prev_a)
                # * state increase in size by 1 due to adding previous action in choose_action, but it will get overwritten in the next iteration
                previous_action = action
            elif policy is not None:
                # fill this in later when you get policy gradient algorithms!
                if policy.type == 'A3C':
                    action, hx = choose_action_a3c(state,policy,hx)
                previous_action = action
            reward, done, _ = env.step(action)
            rewards += reward
            if t >= (t_maxstep - 1):
                done = True
            t += 1

        avgrewards += rewards
    if env.envID == 'tiger':
        if managed == 1:
            print('management was done')
        if surveyed == 1:
            print('survey was done')
    elif env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6'] and drqn == True:
        actiondist = actiondist/np.sum(actiondist)
        print(actiondist)
    return avgrewards/episodenum