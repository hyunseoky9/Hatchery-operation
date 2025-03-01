from setup_logger import setup_logger
import time
import shutil
import subprocess
import os
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import pickle
import numpy as np
import random
from RQNN import *
from RQNN2 import *
from nq import *
from distributionalRL import *
from calc_performance import *
from choose_action import *
from absorbing import *
from pretrain import *
def DRQN(env, paramdf, meta):
    '''
     train using Deep Q Network
     env: environment class object
     num_episodes: number of episodes to train 
     epdecayopt: epsilon decay option
    '''
    # some path and logging settings 
    ## roll otu meta info
    paramid = meta['paramid']
    iteration = meta['iteration']
    seed = meta['seed'] 
    ## Define the path for the new directory
    parent_directory = './DRQN results'
    new_directory = f'seed{seed}_paramset{paramid}'
    path = os.path.join(parent_directory, new_directory)
    ## set path
    os.makedirs(path, exist_ok=True)
    testwd = f'./DRQN results/{new_directory}'
    logger = setup_logger(testwd) ## set up logging
    print(f'paramID: {paramid}, iteration: {iteration}, seed: {seed}')

    # device for pytorch neural network
    #device = (
    #"cuda"
    #if torch.cuda.is_available()
    #else "mps"
    #if torch.backends.mps.is_available()
    #else "cpu"
    #)
    device = 'cpu'
    print(f"Using {device} device")

    # parameters
    ## NN parameters
    # RDQN
    if env.partial == False:
        state_size = len(env.statespace_dim)
    else:
        state_size = len(env.obsspace_dim)
    action_size = env.actionspace_dim[0]
    hidden_size = int(paramdf['hidden_size']) # number of neurons per hidden layer
    hidden_num = int(paramdf['hidden_num']) # number of hidden layers
    lstm_num = int(paramdf['lstm_num']) # number of cells in a lstm layer
    lstm_layers = int(paramdf['lstm_layers']) # number of lstm layers
    ## memory parameters
    memory_size = int(paramdf['memory size']) # memory capacity
    batch_size = int(paramdf['batch size']) # mini-batch size
    seql = int(paramdf['seql']) # sequence length for LSTM. each mini-batch has batch_size number of sequences
    min_seql = int(paramdf['min_seql']) # minimum sequence length for training sequence.
    burninl = int(paramdf['burninl']) # maximmum burn-in length for DRQN
    ## distributional RL atoms size
    Vmin = float(paramdf['vmin'])
    Vmax = float(paramdf['vmax'])
    atomn = int(paramdf['atomn'])


    num_episodes = int(paramdf['episodenum'])
    epdecayparam = eval(paramdf['epsilon'])
    ## etc.
    lr = float(paramdf['lr'])
    lrdecayrate = float(paramdf['lrdecay'])
    if paramdf['minlr'] == 'inf':
        min_lr = float('-inf')
    else:
        min_lr = float(paramdf['minlr'])
    normalize = bool(int(paramdf['normalize']))
    ## extra extensions
    DDQN = bool(int(paramdf['ddqn']))
    nstep = int(paramdf['nstep'])
    distributional = bool(int(paramdf['distributional']))
    ## training cycles
    training_cycle = int(paramdf['training_cycle'])
    target_update_cycle = int(paramdf['target_update_cycle'])
    # performance evaluation
    external_testing = bool(int(paramdf['external testing']))
    # Q initialization
    bestQinit = bool(int(paramdf['bestQinit']))
    # option to input action in the network.
    actioninput = bool(int(paramdf['actioninput']))
    # option to always sample sequences from the start of an episode
    samplefromstart = bool(int(paramdf['samplefromstart']))
    # discount rate
    gamma = env.gamma 
    max_steps = int(paramdf['max_steps']) # max steps per episode
    ## performance testing sample size
    performance_sampleN = int(paramdf['performance_sampleN'])
    final_performance_sampleN = int(paramdf['final_performance_sampleN'])
    evaluation_interval = int(paramdf['evaluation_interval'])
    ## number of steps to run in the absorbing state before terminating the episode
    postterm_len = int(paramdf['postterm_len']) 
    ## actioninput
    actioninputsize = int(actioninput)*len(env.actionspace_dim)


    ## normalization parameters
    if env.contstate == False:
        if env.partial == True:
            state_max = (torch.tensor(env.obsspace_dim, dtype=torch.float32) - 1).to(device)
            state_min = torch.zeros([len(env.obsspace_dim)], dtype=torch.float32).to(device)
        else:
            state_max = (torch.tensor(env.statespace_dim, dtype=torch.float32) - 1).to(device)
            state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32).to(device)
    else:
        if env.partial == True:
            state_max = torch.tensor([env.observations[key][1] for key in env.states.keys()], dtype=torch.float32).to(device)
            state_min = torch.tensor([env.observations[key][0] for key in env.states.keys()], dtype=torch.float32).to(device)
        else:
            state_max = torch.tensor([env.states[key][1] for key in env.states.keys()], dtype=torch.float32).to(device)
            state_min = torch.tensor([env.states[key][0] for key in env.states.keys()], dtype=torch.float32).to(device)
    # append action input
    if actioninput:
        input_max = torch.cat((state_max,torch.ones(actioninputsize)*(np.array(env.actionspace_dim)-1)),0)
        input_min = torch.cat((state_min,torch.zeros(actioninputsize)),0)
    else:
        input_max = state_max
        input_min = state_min

    # initialization
    ## print out extension feature usage
    if distributional:
        print(f'Vmin: {Vmin}, Vmax: {Vmax}, atom N: {atomn}')
    print(f'state size: {state_size}, hidden num: {hidden_num}, hidden size: {hidden_size}, lstm num: {lstm_num}, lstm layers: {lstm_layers}')
    print(f'batch size: {batch_size}, sequence length: {seql}, burn-in length: {burninl}, min sequence length: {min_seql}')
    print(f'action as input: {actioninput}, sample from start: {samplefromstart}')
    print(f'lr: {lr}, lrdecayrate: {lrdecayrate}, min_lr: {min_lr}')
    print(f'max steps: {max_steps}, gamma: {gamma}')
    ## initialize NN
    Q = RQNN(state_size+actioninputsize, action_size, hidden_size, hidden_num, lstm_num, lstm_layers, batch_size, seql, lr, input_min,
              input_max, lrdecayrate, distributional, atomn, Vmin, Vmax, normalize).to(device)
    Q_target = RQNN(state_size+actioninputsize, action_size, hidden_size, hidden_num, lstm_num, lstm_layers, batch_size, seql, lr, input_min,
              input_max, lrdecayrate, distributional, atomn, Vmin, Vmax, normalize).to(device)
    if bestQinit:
        # initialize Q with the best Q network from the previous run
        with open(f"DRQN results/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN.pt", "rb") as file:
            initQ = torch.load(file,weights_only=False)
        print('initializing Q with the best Q network from the previous run')
        Q.load_state_dict(initQ.state_dict())
        initperform = calc_performance(env,device,seed,Q,None,performance_sampleN,max_steps,actioninput) # initial Q's performance
        print(f'performance of the initial Q network: {initperform}')
    else:
        initperform = -100000000

    Q_target.load_state_dict(Q.state_dict())  # Copy weights from Q to Q_target
    Q_target.eval()  # Set target network to evaluation mode (no gradient updates)
    
    ## Define the path for the new directory
    parent_directory = './DRQN results'
    new_directory = f'seed{seed}_paramset{paramid}'
    path = os.path.join(parent_directory, new_directory)

    # run testing script in a separate process if external testing is on
    if external_testing:
        # run the testing script in a separate process
        # Define the script and arguments
        script_name = "performance_tester.py"
        args = ["--num_episodes", f"{num_episodes}", "--DQNorPolicy", "0", "--envID", f"{env.envID}",
                 "--parset", f"{env.parset+1}", "--discset", f"{env.discset}", "--midsample", f"{performance_sampleN}",
                 "--finalsample", f"{final_performance_sampleN}","--initQperformance", f"{initperform}","--maxstep", f"{max_steps}","--drqn", "1",
                 "--actioninput", f"{int(actioninput)}"]
        # Run the script independently with arguments
        #subprocess.Popen(["python", script_name] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["python", script_name] + args)
    
    ## intialize nstep queue
    nq = Nstepqueue(nstep, gamma)
    ## initialize memory
    memory = Memory(memory_size, state_size, len(env.actionspace_dim))
    pretrainsize = 100 #batch_size*(seql+burninl)
    pretrain(env,nq,memory,max_steps,pretrainsize,0,0,postterm_len,1) # prepopulate memory
    print(f'Pretraining memory with {pretrainsize} experiences (buffer size: {memory_size})')

    ## state initialization setting 
    if env.envID == 'Env1.0':
        initlist = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
        reachable_states = torch.tensor([env._unflatten(i[0]) for i in reachables], dtype=torch.float32).to(device) # states extracted from reachable state-action pairs. *there are redundant states on purpose*
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64).to(device)
        reachable_actions = torch.tensor([i[1] for i in reachables], dtype=torch.int64).unsqueeze(1).to(device)
    elif env.envID == 'Env1.1':
        initlist = [-1,-1,-1,-1,-1,-1]
    elif env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','tiger']:
        initlist = [-1,-1,-1,-1,-1,-1]

    ## initialize performance metrics
    # initialize reward performance receptacle
    avgperformances = [] # average of rewards over 100 episodes with policy following trained Q
    final_avgreward = 0
    print(f'performance sampling: {performance_sampleN}/{final_performance_sampleN}')

    # initialize counters
    j = 0 # training cycle counter
    i = 0 # peisode num
    print('-----------------------------------------------')
    # run through the episodes
    done_before = 0
    while i < num_episodes: #delta > theta:
        # update epsilon
        ep = epsilon_update(i,epdecayparam,num_episodes) 
        # initialize state that doesn't start from terminal
        env.reset(initlist) # random initialization
        if env.partial == False:
            S = env.state
        else:
            S = env.obs
        previous_a = 0
        online_hidden = None # hidden state for simulation
        done = False
        t = 0 # timestep num
        termination_t = 0 
        while done == False:
            prev_a = previous_a if actioninput else None
            if t > 0:
                a, online_hidden = choose_action(S, Q, ep, action_size,distributional,device,True,online_hidden,prev_a)
            else:
                a = random.randint(0, action_size-1) # first action in the episode is random for added exploration
                # even if taking random action, run the network anyway to get the update on hidden state.
                _, online_hidden = choose_action(S, Q, ep, action_size,distributional,device,True,online_hidden,prev_a)
            true_state = env.state
            reward, done, _ = env.step(a) # take a step
            if env.episodic == False and env.absorbing_cut == True: # if continuous task
                if absorbing(env,true_state) == True:
                    termination_t += 1
                    if termination_t >= postterm_len: # run x steps once in absorbing state and then terminate
                        done = True
            if t >= max_steps: # finish episode if max steps reached even if terminal state not reached
                done = True
            if env.partial == False:
                nq.add(S, a, reward, env.state, done, previous_a, memory, per=0) # add transition to queue
                S = env.state #  update state
            else:
                nq.add(S, a, reward, env.obs, done, previous_a, memory, per=0)
                S = env.obs
            previous_a = a 
            # train network
            if j % training_cycle == 0:
                # Sample mini-batch from memory
                states, actions, rewards, next_states, dones, previous_actions, burnin_lens, training_lens, total_lens = memory.sample(batch_size,seql, min_seql, burninl, samplefromstart)

                if actioninput: # add previous action in the input
                    states = torch.cat((states, previous_actions), dim=-1)
                    next_states = torch.cat((next_states, actions.float()), dim=-1)

                # Train network
                # Set target_Qs to 0 for states where episode ends
                episode_ends = np.where(dones == True)
                with torch.no_grad():
                    target_Qs, _ = Q_target(next_states, training=True, hidden=None, lengths=total_lens)
                if DDQN:
                    if distributional:
                        with torch.no_grad():
                            action_Qs = Q(next_states)
                        next_EQ = torch.sum(action_Qs * Q.z, dim=-1)  # Expected Q-values for each action
                        best_actions = torch.argmax(next_EQ, dim=-1).unsqueeze(1)  # Best action                        
                        targets = compute_target_distribution(rewards, dones, gamma, nstep, target_Qs, best_actions, Q.z, atomn, Vmin, Vmax)
                    else:
                        if dones.any():
                            target_Qs[episode_ends] = torch.zeros(action_size, device=device)
                        with torch.no_grad():
                            action_Qs, _ = Q(next_states, training=True, hidden=None, lengths=total_lens)
                        next_actions = torch.argmax(action_Qs, dim=2)
                        targets = rewards[:,:target_Qs.shape[1]] + (gamma**nstep) * target_Qs.gather(2, next_actions.unsqueeze(2)).squeeze(2)
                else:
                    if distributional:
                        next_EQ = torch.sum(target_Qs * Q.z, dim=-1)  # Expected Q-values for each action
                        best_actions = torch.argmax(next_EQ, dim=-1).unsqueeze(1)  # Best action
                        targets = compute_target_distribution(rewards, dones, gamma, nstep, target_Qs, best_actions, Q.z, atomn, Vmin, Vmax)
                    else:
                        if dones.any():
                            target_Qs[episode_ends] = torch.zeros(action_size, device=device)
                        targets = rewards[:,:target_Qs.shape[1]] + (gamma**nstep) * torch.max(target_Qs, dim=2)[0]
                td_error = train_model(Q, states, actions, targets, device, total_lens, burnin_lens)

            # update target network
            if j % target_update_cycle == 0:
                Q_target.load_state_dict(Q.state_dict())
                
            t += 1 # update timestep
            j += 1 # update training cycle
        # Decay the learning rate
        Q.scheduler.step() 
        if Q.optimizer.param_groups[0]['lr'] < min_lr:
            Q.optimizer.param_groups[0]['lr'] = min_lr

        if i % evaluation_interval == 0: # calculate average reward every 1000 episodes
            if not external_testing:
                avgperformance = calc_performance(env,device,seed,Q,None,performance_sampleN,max_steps,True,actioninput)
                avgperformances.append(avgperformance)
            if env.envID in ['Env1.0', 'Env1.1', 'Env2.0', 'Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','tiger']:
                torch.save(Q, f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN_episode{i}.pt")

        if i % evaluation_interval == 0: # print outs
            current_lr = Q.optimizer.param_groups[0]['lr']
            if external_testing:
                avgperformance = 'using external testing'
            print(f"Episode {i}, seed {seed}, Learning Rate: {current_lr}, Epsilon: {ep}, Avg Performance: {avgperformance}")
        i += 1 # update episode number

    # calculate final average reward
    if external_testing == False:
        final_avgreward = calc_performance(env,device,seed,Q,None,final_performance_sampleN,max_steps,True,actioninput)
        avgperformances.append(final_avgreward)
        print(f'final average reward (seed {seed}): {final_avgreward}')
    torch.save(Q, f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN_episode{i}.pt")

    # save results and performance metrics.
    ## save last model and the best model (in terms of rewards)
    if env.envID in ['Env1.0','Env1.1','Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','tiger']:
        # last model
        torch.save(Q, f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN.pt")
        # best model
        if not external_testing:
            if max(avgperformances) > initperform:
                bestidx = np.array(avgperformances).argmax()
                bestfilename = f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN_episode{bestidx*evaluation_interval}.pt"
                print(f'best Q network found at episode {bestidx*evaluation_interval}')
                shutil.copy(bestfilename, f"{testwd}/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN.pt")
            else:
                print(f'no improvement in the performance from training')

    ## save performance
    if external_testing == False:
        np.save(f"{testwd}/rewards_{env.envID}_par{env.parset}_dis{env.discset}_DRQN.npy", avgperformances)

    ## lastly save the configuration.
    param_file_path = os.path.join(testwd, f"config.txt")
    with open(param_file_path, 'w') as param_file:
        for key, value in paramdf.items():
            param_file.write(f"{key}: {value}\n")

    return avgperformances, final_avgreward

def _make_discrete_Q(Q,env,device):
    # make a discrete Q table
    if env.envID in ['Env1.0']:
        states = torch.tensor([env._unflatten(i) for i in range(np.prod(env.statespace_dim))], dtype=torch.float32)
        Q_discrete = np.zeros((np.prod(env.statespace_dim),len(env.actions["a"])))
        for i in range(np.prod(env.statespace_dim)):
            if Q.distributional:
                Q_expected = torch.sum(Q(states[i].unsqueeze(0)) * Q.z, dim=-1) # sum over atoms for each action
                Q_discrete[i,:] = Q_expected.detach().cpu().numpy()
            else:
                Q_discrete[i,:] = Q(states[i].unsqueeze(0)).detach().cpu().numpy()
    return Q_discrete


class Memory():
    def __init__(self, max_size, state_dim, action_dim):
        # Preallocate memory
        self.states_buffer = np.ones((max_size, state_dim), dtype=np.float32)*-2
        self.actions_buffer = np.ones((max_size, action_dim), dtype=np.float32)*-2
        self.rewards_buffer = np.ones(max_size, dtype=np.float32)*-2
        self.next_states_buffer = np.ones((max_size, state_dim), dtype=np.float32)*-2
        self.done_buffer = np.zeros(max_size, dtype=np.bool_)
        self.previous_actions_buffer = np.ones((max_size, action_dim), dtype=np.float32)*-2
        self.index = 0
        self.size = 0
        self.buffer_size = max_size

    def add(self, state, action, reward, next_state, done, previous_action=None):
        self.states_buffer[self.index] = state
        self.actions_buffer[self.index] = action
        self.rewards_buffer[self.index] = reward
        self.next_states_buffer[self.index] = next_state
        self.done_buffer[self.index] = done
        self.previous_actions_buffer[self.index] = action
        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, seq_len, min_seq_len, burn_in_len, samplefromstart):
        """
        Sample a batch of sequences for DRQN with burn-in:
          - Pick a 'training start' for the L-length portion.
          - Prepend up to 'burn_in_len' transitions before training start.
          - If we hit the beginning of an episode or the memory, we reduce the burn-in portion.
          - If we hit done, we stop and pad if the sequence does not meet seql
        
        The final length of each sequence = (burn_in_part + training_part).
        We'll do *one* combined sequence, but only compute loss on the training part of the sequences.
        
        Returns:
          states, actions, rewards, next_states, done_flags
          each of shape (batch_size, total_seq, dimension of each element (e.g. state, action, reward, etc.))
          where total_seq <= burn_in_len + seq_len (it can be shorter if we hit the end of the episode).
          
          We also return indices to indicate which part is burn-in vs. training, and indices to indicate 
          which part is padding.

        * if samplefromstart == True: always pick the start of the sequence as the start of an episode.
        """
        # Prepare lists for the final batch
        state_dim = self.states_buffer.shape[1]
        action_dim = self.actions_buffer.shape[1]
        totlen = seq_len + burn_in_len
        batch_states = np.ones((batch_size, totlen, state_dim), dtype=np.float32)*(-1)
        batch_actions = np.ones((batch_size, totlen, action_dim), dtype=np.float32)*(-1)
        batch_rewards = np.ones((batch_size, totlen), dtype=np.float32)*(-1)
        batch_next_states = np.ones((batch_size, totlen, state_dim), dtype=np.float32)*(-1)
        batch_previous_actions = np.ones((batch_size, totlen, action_dim), dtype=np.float32)*(-1)
        batch_dones = np.zeros((batch_size, totlen), dtype=bool)
        burnin_lens = []
        total_lens = []
        training_lens = []
        for b in range(batch_size):
            # 1) Gather the training steps, but stop if we see 'done' or end of the memory. Randomly pick a training start index (the first frame to be used for loss).
            # Randomly pick a training start index
            # ensure there's at least min_seq_len number of valid transition from the train_start.
            found_valid_seq = False
            for tries in range(1000):
                if samplefromstart:
                    episode_starts = np.where(self.done_buffer==True)[0] + 1
                    if len(episode_starts) == 0:
                        train_start = 0
                    else:
                        train_start = np.random.choice(episode_starts)
                    if train_start == self.size:
                        train_start = 0
                else:
                    train_start = np.random.randint(0, self.size - min_seq_len + 1)  # random index in [0, size)
                train_indices = list(np.arange(train_start, min(train_start + seq_len, self.size))) # check that train indices don't go over the buffer size

                if (train_start+seq_len) > self.size and self.buffer_size == self.size: # if the last index is the last index of the buffer, then start appending from the beginning of the buffer
                    train_indices += list(np.arange(0, (train_start+seq_len)%self.size))
                donecheck = np.where(self.done_buffer[train_indices]==True) # check if there's a done in the train indices
                if len(donecheck[0]) > 0: # if there are dones in the train indices cut the train indices to the first done.
                    train_indices = train_indices[:donecheck[0][0]+1]
                if len(train_indices) >= min_seq_len:
                    found_valid_seq = True
                    break
            training_lens.append(len(train_indices))
            if not found_valid_seq:
                raise ValueError("SEQUENCE LENGTH ERROR")
                

            # 2) Gather the burnin steps, but stop if we see done or beginning of the memory.
            # We'll gather them in reverse, then flip.
            burnin_indices = list(np.arange(max(train_start-burn_in_len,0),train_start))
            donecheck = np.where(self.done_buffer[burnin_indices]==True)
            if len(donecheck[0]) > 0:
                burnin_indices = burnin_indices[donecheck[0][-1]+1:]
            burnin_lens.append(len(burnin_indices))

            # The combined sequence = burnin + training
            full_indices = burnin_indices + train_indices
            total_lens.append(len(full_indices))

            # We'll store the transitions from these indices

            states_seq = self.states_buffer[full_indices]
            actions_seq = self.actions_buffer[full_indices]
            rewards_seq = self.rewards_buffer[full_indices]
            next_states_seq = self.next_states_buffer[full_indices]
            done_seq = self.done_buffer[full_indices]
            previous_actions_seq = self.previous_actions_buffer[full_indices]
            
            batch_states[b, :len(full_indices)] = states_seq
            batch_actions[b, :len(full_indices)] = actions_seq
            batch_rewards[b, :len(full_indices)] = rewards_seq
            batch_next_states[b, :len(full_indices)] = next_states_seq
            batch_dones[b, :len(full_indices)] = done_seq
            batch_previous_actions[b, :len(full_indices)] = previous_actions_seq

        # convert them into pytorch tensors
        batch_states = torch.tensor(batch_states, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.int64)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32)
        batch_previous_actions = torch.tensor(batch_previous_actions, dtype=torch.float32)
        # * dones do not need to be converted to pytorch tensors

        # covert lengths into numpy arrays
        burnin_lens = np.array(burnin_lens)
        training_lens = np.array(training_lens)
        total_lens = np.array(total_lens)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_previous_actions, burnin_lens, training_lens, total_lens


def epsilon_update(i,param,num_episodes):
    # update epsilon
    if param['type'] == 0:
        # inverse decay
        return 1/(i+1)
    elif param['type'] == 1:
        # inverse decay with a minimum epsilon of 0.01
        return max(1/(i+1), 0.2)
    elif param['type'] == 2:
        # pure exploration for 10% of the episodes
        if i < num_episodes*0.1:
            return 1
        else:
            return max(1/(i-(np.ceil(num_episodes*0.1)-1)), 0.01)
    elif param['type'] == 3: # exponential decay
        minep = param['minep']
        maxep = param['maxep']
        beta = param['rate']/num_episodes
        newep = minep + (maxep - minep)*np.exp(-beta*i)
        return newep
    elif param['type'] == 4: # logistic decay
        if type(param['fix']) == int:
            fix = param['fix']
        else:
            fix = num_episodes
        a= param['a']
        b=-param['bf']/fix
        c=-fix*param['cf']
        return max(a/(1+np.exp(-b*(i+c))), param['minep'])
    elif param['type'] == 5: # linear decay
        return max(1-i/num_episodes, 0)
    elif param['type'] == 6: # fixed decay
        return param['val']

def train_model(Q, batch_states, batch_actions, batch_targets, device, total_lens, burnin_lens):
    Q.train()
    # Compute predictions
    batch_predictions, _ = Q(batch_states, training=True, hidden=None, lengths=total_lens)

    total_loss = 0.0
    all_td_errors = []
    for batch in range(batch_states.shape[0]):
        states, actions, targets, predictions = batch_states[batch], batch_actions[batch], batch_targets[batch], batch_predictions[batch]
        states, actions, targets, predictions = states.to(device), actions.to(device), targets.to(device), predictions.to(device)
        
        # clip burnin and pads
        actions = actions[burnin_lens[batch]:total_lens[batch]]
        predictions = predictions[burnin_lens[batch]:total_lens[batch]]
        targets = targets[burnin_lens[batch]:total_lens[batch]]
        # compute_loss
        if Q.distributional:
            predictions = predictions.gather(1, actions.unsqueeze(-1).expand(-1,-1,Q.atomn)) # Get Q-values for the selected actions
            predictions = predictions.clamp(min=1e-9) # Avoid log 0
            cross_entropy = -torch.sum(targets * torch.log(predictions), dim=-1)  # Sum over atoms
            loss = cross_entropy.mean()  # Average over batch
            td_errors = cross_entropy.squeeze() # distributional RL uses cross entropy as scalar distance btw target and prediction for priority
        else:
            # loss = Q.loss_fn(predictions, targets) # Compute the loss
            predictions = predictions.gather(1, actions).squeeze(1) # Get Q-values for the selected actions
            td_errors = targets - predictions
            loss = (td_errors ** 2).mean()
        total_loss += loss
        all_td_errors.append(td_errors.detach().cpu())
        

    # Backpropagation
    total_loss = total_loss / batch_states.shape[0]
    total_loss.backward()
    Q.optimizer.step()
    Q.optimizer.zero_grad()
    return all_td_errors

