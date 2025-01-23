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
from nq import *
from distributionalRL import *
from calc_performance import *
from choose_action import *
from absorbing import *
from pretrain import *
def DRQN(env,num_episodes,epdecayopt,
            DDQN,nstep,distributional,
            lrdecayrate,lr,min_lr,
            training_cycle,target_update_cycle, 
            external_testing, normalize, bestQinit, actioninput):
    # train using Deep Q Network
    # env: environment class object
    # num_episodes: number of episodes to train 
    # epdecayopt: epsilon decay option
    
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
    # RDQN
    if env.partial == False:
        state_size = len(env.statespace_dim)
    else:
        state_size = len(env.obsspace_dim)
    action_size = env.actionspace_dim[0]
    hidden_size = 20
    hidden_num = 3
    lstm_num = 20
    ## memory parameters
    memory_size = 1000 # memory capacity
    batch_size = 4 # mini-batch size
    seql = 6 # sequence length for LSTM. each mini-batch has batch_size number of sequences
    min_seql = 3 # minimum sequence length for training sequence.
    burninl = 5 # maximmum burn-in length for DRQN
    ## distributional RL atoms size
    Vmin = -104
    Vmax = 1002
    atomn = 32
    ## actioninput
    actioninputsize = int(actioninput)*len(env.actionspace_dim)

    ## etc.
    #lr = 0.01
    #min_lr = 1e-6
    gamma = env.gamma # discount rate
    max_steps = 100 # max steps per episode
    ## cycles
    #training_cycle = 7 # number of steps where the network is trained
    #target_update_cycle = 10 # number of steps where the target network is updated
    ## performance testing sample size
    performance_sampleN = 1000
    final_performance_sampleN = 1000

    ## normalization parameters
    if env.envID == 'Env1.0': # for discrete states
        state_max = (torch.tensor(np.array(env.statespace_dim)-1, dtype=torch.float32) - 1).to(device)
        state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32).to(device)
    elif env.envID in ['Env1.1','Env1.2']: # for continuous states
        state_max = torch.tensor([env.states[key][1] for key in env.states.keys()], dtype=torch.float32).to(device)
        state_min = torch.tensor([env.states[key][0] for key in env.states.keys()], dtype=torch.float32).to(device)
    elif env.envID == 'Env2.0': # state is really observation in env2.0. We'll call the actual states as hidden states. This is done to make the code consistent with env1.0
        state_max = (torch.tensor(np.array(env.obsspace_dim)-1, dtype=torch.float32)).to(device)
        state_min = (torch.zeros([len(env.obsspace_dim)], dtype=torch.float32)).to(device) 
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
    print(f'lr: {lr}, lrdecayrate: {lrdecayrate}, min_lr: {min_lr}')
    ## initialize NN
    Q = RQNN(state_size+actioninputsize, action_size, hidden_size, hidden_num, lstm_num, batch_size, seql, lr, input_min,
              input_max, lrdecayrate, distributional, atomn, Vmin, Vmax, normalize).to(device)
    Q_target = RQNN(state_size+actioninputsize, action_size, hidden_size, hidden_num, lstm_num, batch_size, seql, lr, input_min,
              input_max, lrdecayrate, distributional, atomn, Vmin, Vmax, normalize).to(device)
    if bestQinit:
        # initialize Q with the best Q network from the previous run
        with open(f"DRQN results/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DRQN.pt", "rb") as file:
            initQ = torch.load(file,weights_only=False)
        print('initializing Q with the best Q network from the previous run')
        Q.load_state_dict(initQ.state_dict())
        initperform = calc_performance(env,device,Q,None,performance_sampleN) # initial Q's performance
        print(f'performance of the initial Q network: {initperform}')
    else:
        initperform = -100000000

    Q_target.load_state_dict(Q.state_dict())  # Copy weights from Q to Q_target
    Q_target.eval()  # Set target network to evaluation mode (no gradient updates)
    

   ## start testing process
    testwd = './DRQN results/intermediate training Q network'
    # delete all the previous network files in the intermediate network folder to not test the old Q networks
    for file in os.listdir(testwd):
        try:
            os.remove(os.path.join(testwd,file))
        except PermissionError:
            print(f"File {testwd} is locked. Retrying...")
            time.sleep(5)  # Wait 5 second
            os.remove(testwd)  # Retry deletion
    # run testing script in a separate process if external testing is on
    if external_testing:
        # run the testing script in a separate process
        # Define the script and arguments
        script_name = "performance_tester.py"
        args = ["--num_episodes", f"{num_episodes}", "--DQNorPolicy", "0", "--envID", f"{env.envID}",
                 "--parset", f"{env.parset+1}", "--discset", f"{env.discset}", "--midsample", f"{performance_sampleN}",
                 "--finalsample", f"{final_performance_sampleN}","--initQperformance", f"{initperform}"]
        # Run the script independently with arguments
        #subprocess.Popen(["python", script_name] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["python", script_name] + args)
    
    ## intialize nstep queue
    nq = Nstepqueue(nstep, gamma)
    ## initialize memory
    memory = Memory(memory_size, state_size, len(env.actionspace_dim))
    pretrain(env,nq,memory,max_steps,batch_size*(seql+burninl),PrioritizedReplay=0,max_priority=0) # prepopulate memory
    print(f'Pretraining memory with {batch_size*(seql+burninl)} experiences (buffer size: {memory_size})')

    ## state initialization setting 
    if env.envID == 'Env1.0':
        initlist = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
        reachable_states = torch.tensor([env._unflatten(i[0]) for i in reachables], dtype=torch.float32).to(device) # states extracted from reachable state-action pairs. *there are redundant states on purpose*
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64).to(device)
        reachable_actions = torch.tensor([i[1] for i in reachables], dtype=torch.int64).unsqueeze(1).to(device)
    elif env.envID == 'Env1.1':
        initlist = [-1,-1,-1,-1,-1,-1]
    elif env.envID == 'Env2.0':
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
    while i < num_episodes: #delta > theta:
        # update epsilon
        ep = epsilon_update(i,epdecayopt,num_episodes) 
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
        
        while done == False:    
            if t > 0:
                a, online_hidden = choose_action(S, Q, ep, action_size,distributional,device,True,online_hidden)
            else:
                a = random.randint(0, action_size-1) # first action in the episode is random for added exploration
                # even if taking random action, run the network anyway to get the update on hidden state.
                _, online_hidden = choose_action(S, Q, ep, action_size,distributional,device,True,online_hidden)

            reward, done, _ = env.step(a) # take a step
            if env.episodic == False and env.absorbing_cut == True: # if continuous task
                if absorbing(env,S) == True:
                    termination_t += 1
                    if termination_t >= 5:
                        done = True
            if t >= max_steps: # finish episode if max steps reached even if terminal state not reached
                done = True

            if env.partial == False:
                nq.add(S, a, reward, env.state, previous_a, done, memory, per=0) # add transition to queue
                S = env.state #  update state
            else:
                nq.add(S, a, reward, env.obs, previous_a, done, memory, per=0)
                S = env.obs
            previous_a = a 
            # train network
            if j % training_cycle == 0:
                # Sample mini-batch from memory
                memory.sample(batch_size,seql, min_seql, burninl)
                states, actions, rewards, next_states, dones, previous_actions = memory.sample(batch_size)
                weights = np.ones(batch_size)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                previous_actions = torch.tensor(previous_actions, dtype=torch.float32).to(device)
                weights = torch.tensor(weights, dtype=torch.float32).to(device)
                # Train network

                # Set target_Qs to 0 for states where episode ends
                episode_ends = np.where(dones == True)[0]
                target_Qs = Q_target(next_states)
                if DDQN:
                    if distributional:
                        next_EQ = torch.sum(target_Qs * Q.z, dim=-1)  # Expected Q-values for each action
                        best_actions = torch.argmax(next_EQ, dim=-1).unsqueeze(1)  # Best action                        
                        targets = compute_target_distribution(rewards, dones, gamma, nstep, target_Qs, best_actions, Q.z, atomn, Vmin, Vmax)                        
                    else:
                        if dones.any():
                            target_Qs[episode_ends] = torch.zeros(action_size, device=device)
                        next_actions = torch.argmax(Q(next_states), dim=1)
                        targets = rewards + (gamma**nstep) * target_Qs.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    if distributional:
                        next_EQ = torch.sum(target_Qs * Q.z, dim=-1)  # Expected Q-values for each action
                        best_actions = torch.argmax(next_EQ, dim=-1).unsqueeze(1)  # Best action
                        targets = compute_target_distribution(rewards, dones, gamma, nstep, target_Qs, best_actions, Q.z, atomn, Vmin, Vmax)
                    else:
                        if dones.any():
                            target_Qs[episode_ends] = torch.zeros(action_size, device=device)
                        targets = rewards + (gamma**nstep) * torch.max(target_Qs, dim=1)[0]
                td_error = train_model(Q, [(states, actions, targets)], weights, device)

            # update target network
            if j % target_update_cycle == 0:
                Q_target.load_state_dict(Q.state_dict())
                
            t += 1 # update timestep
            j += 1 # update training cycle
        # Decay the learning rate
        Q.scheduler.step() 
        if Q.optimizer.param_groups[0]['lr'] < min_lr:
            Q.optimizer.param_groups[0]['lr'] = min_lr

        if i % 1000 == 0: # calculate average reward every 1000 episodes
            if not external_testing:
                avgperformance = calc_performance(env,device,Q,None,performance_sampleN)
                avgperformances.append(avgperformance)
            if env.envID in ['Env1.0', 'Env1.1']:
                torch.save(Q, f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN_episode{i}.pt")

        if i % 1000 == 0: # print outs
            current_lr = Q.optimizer.param_groups[0]['lr']
            if external_testing:
                avgperformance = 'using external testing'
            print(f"Episode {i}, Learning Rate: {current_lr} Avg Performance: {avgperformance}")
        i += 1 # update episode number

    # calculate final average reward
    print('calculating the average reward with the final Q network')
    if external_testing == False:
        final_avgreward = calc_performance(env,device,Q,None,final_performance_sampleN)
        avgperformances.append(final_avgreward)
        print(f'final average reward: {final_avgreward}')
    torch.save(Q, f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN_episode{i}.pt")

    # save results and performance metrics.
    ## save last model and the best model (in terms of rewards)
    if env.envID in ['Env1.0','Env1.1']:
        # last model
        wd = './deepQN results'
        torch.save(Q, f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pt")
        # best model
        if not external_testing:
            if max(avgperformances) > initperform:
                bestidx = np.array(avgperformances).argmax()
                bestfilename = f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN_episode{bestidx*1000}.pt"
                shutil.copy(bestfilename, f"{wd}/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pt")
            else:
                print(f'no improvement in the performance from training')

    ## make a discrete Q table if the environment is discrete and save it
    if env.envID == 'Env1.0':
        Q_discrete = _make_discrete_Q(Q,env,device)
        policy = _get_policy(env,Q_discrete)
        wd = './deepQN results'
        with open(f"{wd}/Q_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pkl", "wb") as file:
            pickle.dump(Q_discrete, file)
        with open(f"{wd}/policy_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pkl", "wb") as file:
            pickle.dump(policy, file)
    else:
        Q_discrete = None
        policy = None
    ## save performance
    if external_testing == False:
        np.save(f"{wd}/rewards_{env.envID}_par{env.parset}_dis{env.discset}_DQN.npy", avgperformances)
    return Q_discrete, policy, avgperformances, final_avgreward

def _make_discrete_Q(Q,env,device):
    # make a discrete Q table
    if env.envID == 'Env1.0':
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

    def sample(self, batch_size, seq_len, min_seq_len, burn_in_len):
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
            for tries in range(100):
                train_start = np.random.randint(0, self.size)  # random index in [0, size)
                train_indices = list(np.arange(train_start, min(train_start + seq_len, self.size))) # check that train indices don't go over the buffer size
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
        batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32)
        # * dones do not need to be converted to pytorch tensors

        # covert lengths into numpy arrays
        burnin_lens = np.array(burnin_lens)
        training_lens = np.array(training_lens)
        total_lens = np.array(total_lens)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, burnin_lens, training_lens, total_lens


def epsilon_update(i,option,num_episodes):
    # update epsilon
    if option == 0:
        # inverse decay
        return 1/(i+1)
    elif option == 1:
        # inverse decay with a minimum epsilon of 0.01
        return max(1/(i+1), 0.2)
    elif option == 2:
        # pure exploration for 10% of the episodes
        if i < num_episodes*0.1:
            return 1
        else:
            return max(1/(i-(np.ceil(num_episodes*0.1)-1)), 0.01)
    elif option == 3: # exponential decay
        a = 1/num_episodes*10
        return np.exp(-a*i)
    elif option == 4: # logistic decay
        fix = 100000
        a=0.1
        b=-10*1/fix*3
        c=-fix*0.4
        return max(a/(1+np.exp(-b*(i+c))), 0.01)

def _get_policy(env,Q):
    # get policy from Q function
    policy = np.zeros(np.prod(env.statespace_dim))
    for i in range(np.prod(env.statespace_dim)):
        policy[i] = np.argmax(Q[i,:])
    return policy

def train_model(Q, data, weights, device):
    Q.train()
    for batch, (states, actions, targets) in enumerate(data):
        states, actions, targets = states.to(device), actions.to(device), targets.to(device)

        # Compute predictions
        predictions = Q(states) 
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
            loss = (weights * (td_errors ** 2)).mean()

        # Backpropagation
        loss.backward()
        Q.optimizer.step()
        Q.optimizer.zero_grad()
        return td_errors

def compute_loss(Q, states, actions, targetQs): 
    """
    Compute the loss and perform a backward pass.
    
    Parameters:
        states (torch.Tensor): Input states.
        actions (torch.Tensor): Actions taken (as indices).
        targetQs (torch.Tensor): Target Q values.
    """
    q_values = Q(states) # Forward pass
    if Q.distributional:
        # Compute expected Q-values from the distribution
        q_expected = torch.sum(q_values * Q.z, dim=-1)  # Expected Q-values for each action
        selected_q_values = q_expected.gather(1, actions).squeeze(1)  # Select Q-values for given actions

        # Compute MSE loss with target Q values
        loss = ((selected_q_values - targetQs) ** 2).mean()  # Compute MSE manually
    else:
        selected_q_values = q_values.gather(1, actions).squeeze(1) # Get Q-values for the selected actions
        loss = Q.loss_fn(selected_q_values, targetQs) # Compute the loss
    return loss