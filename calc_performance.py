from call_paramset import call_paramset, call_env
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from choose_action import choose_action
from choose_action_a3c import choose_action_a3c
def calc_performance(env, device, seed, Q=None, policy=None, episodenum=1000, t_maxstep=1000, drqn=False, actioninput=False):
    """
    parallelized version
    calculate the performance of the agent in the environment.
    For DQN calculate performance with the Q network. (use Q variable in the function input)
    For Tabular Q learning and value iteration, calculate perofrmance using the policy table.
    For policy gradient methods, calculate performance using the policy network.
    """
    print('parallel calc_performance called')
    action_size = env.actionspace_dim[0]
    if Q is not None:
        distributional = Q.distributional
    fstack = 1 if not hasattr(Q, 'fstack') else Q.fstack
    # for getting policy related info
    policy_info = {
    'managed': mp.Value('i', 0),
    'surveyed': mp.Value('i', 0),
    'actiondist': torch.zeros(action_size).share_memory_(),  # Shared memory array
    'lock': mp.Lock()  # Lock to ensure thread safety
    }
    # set environment configuration for the workers
    initlist = None
    config = "{'init': %s,'paramset': %d, 'discretization': %d}" %(str(initlist),env.parset+1,env.discset)
    envinit_params = {'envconfig': config, 'envid': env.envID}

    # worker parameters
    #################
    num_workers = 4
    #################
    if Q is not None:
        Q.share_memory()
    if policy is not None:
        policy.share_memory()
    total_rewards = mp.Value('f', 0.0)


    # Set multiprocessing method
    mp.set_start_method("spawn", force=True)
    # start timer
    processes = []
    # divide the number of epsiodes by workers.
    episodenum_perworker = [int(np.floor(episodenum/num_workers))]*(num_workers - 1)
    episodenum_perworker = episodenum_perworker + [int(episodenum - np.sum(episodenum_perworker))]
    # Spawn worker processes
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(Q, policy, episodenum, episodenum_perworker[worker_id], worker_id, envinit_params, t_maxstep, drqn, actioninput, fstack, action_size, distributional, device, policy_info, total_rewards,seed))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()

    if env.envID == 'tiger':
        if policy_info['managed'] == 1:
            print('management was done')
        if policy_info['surveyed'] == 1:
            print('survey was done')
    elif env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','Env2.7'] and drqn == True:
        policy_info['actiondist'].div_(policy_info['actiondist'].sum())
        print(policy_info['actiondist'])
    
    return total_rewards.value/episodenum




def worker(Q, policy, episodenum, workerepisodenum, worker_id, envinit_params, t_maxstep, drqn, actioninput, fstack, action_size, distributional, device, policy_info, total_rewards,seed):
    # set seed
    worker_seed = seed + (worker_id + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # set env
    env = call_env(envinit_params)

    totrewards_perworker = 0 # set reward counter.

    if env.envID == 'tiger':
        managed = 0 # for tiger environment
        surveyed = 0 # for tiger environment
    if env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6']:
        actiondist = torch.zeros(action_size, dtype=torch.float32)

    for i in range(workerepisodenum):
        rewards = 0
        env.reset()
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
                stack = stack[len(env.state):] + env.state
            else:
                state = env.obs
                stack = stack[len(env.obs):] + env.obs

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
        totrewards_perworker += rewards

    # additional policy related info update
    with policy_info['lock']:  
        if env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6']:
            policy_info['actiondist'].add_(actiondist) # Directly update shared tensor\
        if env.envID == 'tiger':
            if policy_info['surveyed'].value == 0:
                policy_info['surveyed'].value = surveyed
            if policy_info['managed'].value == 0:
                policy_info['managed'].value = managed
        
    # upload total rewards    
    with total_rewards.get_lock(): 
        total_rewards.value += totrewards_perworker # update global counter

    