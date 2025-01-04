import subprocess
import os
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR
import pickle
import numpy as np
import random
from QNN import QNN
from DuelQNN import DuelQNN
from PrioritizedMemory import *
from nq import *
from distributionalRL import *
from calc_performance import *

def Rainbow(env,num_episodes,epdecayopt,
            DDQN,DuelingDQN,PrioritizedReplay,nstep,noisy,distributional,
            lrdecayrate,lr,min_lr,
            training_cycle,target_update_cycle,
            calc_MSE, external_testing, normalize):
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
    # DQN
    state_size = len(env.statespace_dim)
    action_size = env.actionspace_dim[0]
    hidden_size = 20
    hidden_num = 3
    # Dueling DQN
    hidden_num_shared = 1
    hidden_num_split = 1
    hidden_size_shared = 30
    hidden_size_split = 20
    # Prioritized Replay
    alpha = 0.4 # priority importance
    beta0 = 0.4 # initial beta
    per_epsilon = 1e-6 # small value to avoid zero priority
    max_abstd = 1 # initial max priority
    ## memory parameters
    memory_size = 1000 # memory capacity
    batch_size = 100 # experience mini-batch size
    ## distributional RL atoms size
    Vmin = -104
    Vmax = 70
    atomn = 16

    ## etc.
    #lr = 0.01
    #min_lr = 1e-6
    gamma = env.gamma # discount rate
    max_steps = 1000 # max steps per episode
    ## cycles
    #training_cycle = 7 # number of steps where the network is trained
    #target_update_cycle = 10 # number of steps where the target network is updated
    ## performance testing sample size
    performance_sampleN = 100
    final_performance_sampleN = 100

    ## testing settings
    testwd = './deepQN results/intermediate training Q network'
    # delete all the previous network files in the intermediate network folder to not test the old Q networks
    for file in os.listdir(testwd):
        os.remove(os.path.join(testwd,file))

    if external_testing:
        # run the testing script in a separate process
        # Define the script and arguments
        script_name = "performance_tester.py"
        args = ["--num_episodes", f"{num_episodes}", "--DQNorPolicy", "0", "--envID", f"{env.envID}",
                 "--parset", f"{env.parset+1}", "--discset", f"{env.discset}", "--midsample", f"{performance_sampleN}",
                 "--finalsample", f"{final_performance_sampleN}"]
        # Run the script independently with arguments
        #subprocess.Popen(["python", script_name] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["python", script_name] + args)

    ## normalization parameters
    if env.envID == 'Env1.0': # for discrete states
        state_max = (torch.tensor(env.statespace_dim, dtype=torch.float32) - 1).to(device)
        state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32).to(device)
    elif env.envID in ['Env1.1','Env1.2']: # for continuous states
        state_max = torch.tensor([env.states[key][1] for key in env.states.keys()], dtype=torch.float32).to(device)
        state_min = torch.tensor([env.states[key][0] for key in env.states.keys()], dtype=torch.float32).to(device)
        
    # initialization
    ## print out extension feature usage
    print(f'DuelingDQN: {DuelingDQN}\nDDQN: {DDQN}\nPrioritizedReplay: {PrioritizedReplay}\nnstep: {nstep}\nnoisynet: {noisy}\ndistributional RL: {distributional}')
    if DuelingDQN:
        print(f'hidden_size_shared: {hidden_size_shared}, hidden_size_split: {hidden_size_split}, hidden_num_shared: {hidden_num_shared}, hidden_num_split: {hidden_num_split}')
    else:
        print(f'hidden_size: {hidden_size}, hidden_num: {hidden_num}')
    if PrioritizedReplay:
        print(f'alpha: {alpha}, beta0: {beta0}, per_epsilon: {per_epsilon}')
    if distributional:
        print(f'Vmin: {Vmin}, Vmax: {Vmax}, atom N: {atomn}')
    print(f'lr: {lr}, lrdecayrate: {lrdecayrate}, min_lr: {min_lr}')
    ## initialize NN
    if DuelingDQN:
        Q = DuelQNN(state_size, action_size, hidden_size_shared, hidden_size_split, hidden_num_shared,
                     hidden_num_split, lr, state_min, state_max,lrdecayrate,noisy,distributional,atomn, Vmin, Vmax, normalize).to(device)
        Q_target = DuelQNN(state_size, action_size, hidden_size_shared, hidden_size_split, hidden_num_shared,
                            hidden_num_split, lr, state_min, state_max,lrdecayrate,noisy,distributional,atomn, Vmin, Vmax, normalize).to(device)
    else:
        Q = QNN(state_size, action_size, hidden_size, hidden_num, lr, state_min, state_max,
                lrdecayrate,noisy, distributional, atomn, Vmin, Vmax, normalize).to(device)
        Q_target = QNN(state_size, action_size, hidden_size, hidden_num, lr, state_min, 
                       state_max,lrdecayrate,noisy, distributional, atomn, Vmin, Vmax, normalize).to(device)
    Q_target.load_state_dict(Q.state_dict())  # Copy weights from Q to Q_target
    Q_target.eval()  # Set target network to evaluation mode (no gradient updates)

    ## intialize nstep queue
    nq = Nstepqueue(nstep, gamma)
    ## initialize memory
    if PrioritizedReplay:
        memory = PMemory(memory_size, alpha, per_epsilon, max_abstd)
        beta = beta0
        pretrain(env,nq,memory,batch_size,PrioritizedReplay,memory.max_abstd) # prepopulate memory
    else:
        memory = Memory(memory_size, state_size, len(env.actionspace_dim))
        pretrain(env,nq,memory,batch_size,PrioritizedReplay,0) # prepopulate memory
    print(f'Pretraining memory with {batch_size} experiences')

    ## state initialization setting 
    if env.envID == 'Env1.0':
        initlist = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
        reachable_states = torch.tensor([env._unflatten(i[0]) for i in reachables], dtype=torch.float32).to(device)
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64).to(device)
        reachable_actions = torch.tensor([i[1] for i in reachables], dtype=torch.int64).unsqueeze(1).to(device)
    elif env.envID == 'Env1.1':
        initlist = [-1,-1,-1,-1,-1,-1]
    
    ## initialize performance metrics
    # load Q function from the value iteration for calculating MSE
    if calc_MSE:
        if env.envID == 'Env1.0':
            with open(f"value iter results/Q_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
                Q_vi = pickle.load(file)
            Q_vi = torch.tensor(Q_vi[reachable_uniquestateid].flatten(), dtype=torch.float32).to(device)
    MSE = []
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
        if noisy: # turn off epsilon greedy for noisy nets
            ep = 0
        else:
            ep = epsilon_update(i,epdecayopt,num_episodes) 
        # initialize state that doesn't start from terminal
        env.reset(initlist) # random initialization
        S = env.state
        done = False
        
        t = 0 # timestep num
        while done == False:    
            if t > 0:
                a = choose_action(S, Q, ep, action_size,distributional,device)
            else:
                a = random.randint(0, action_size-1) # first action in the episode is random for added exploration
            reward, done, _ = env.step(a) # take a step
            nq.add(S, a, reward, env.state, done, memory, PrioritizedReplay) # add transition to queue
            S = env.state #  update state
            if t >= max_steps: # finish episode if max steps reached even if terminal state not reached
                done = True
            # train network
            if j % training_cycle == 0:
                # Sample mini-batch from memory
                if PrioritizedReplay:
                    mini_batch, idxs, weights = memory.sample(batch_size, beta)
                    states, actions, rewards, next_states, dones = zip(*mini_batch)
                    dones = np.array(dones)
                    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                else:
                    states, actions, rewards, next_states, dones = memory.sample(batch_size)
                    weights = np.ones(batch_size)
                    actions = torch.tensor(actions, dtype=torch.int64).to(device)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
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

                # Update priorities
                if PrioritizedReplay:
                    td_error = np.abs(td_error.detach().cpu().numpy())
                    memory.update_priorities(idxs, td_error)
                    memory.max_abstd = max(memory.max_abstd, np.max(td_error))

            # update target network
            if j % target_update_cycle == 0:
                Q_target.load_state_dict(Q.state_dict())
                
            t += 1 # update timestep
            j += 1 # update training cycle
        # beta update for prioritized replay
        if PrioritizedReplay: 
            beta += (1.0 - beta0)/num_episodes
        # Decay the learning rate
        Q.scheduler.step() 
        if Q.optimizer.param_groups[0]['lr'] < min_lr:
            Q.optimizer.param_groups[0]['lr'] = min_lr

        if i % 100 == 0: # MSE calculation
            if calc_MSE:
                mse_value = test_model(Q, reachable_states, reachable_actions, Q_vi, noisy, device)
                MSE.append(mse_value)
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

            if calc_MSE:
                print(f"Episode {i}, Learning Rate: {current_lr} MSE: {round(mse_value,2)} Avg Performance: {avgperformance}")
            else:
                print(f"Episode {i}, Learning Rate: {current_lr} Avg Performance: {avgperformance}")

            meansig = 0
            if noisy:
                if DuelingDQN:
                    for layer in Q.shared_linear_relu_stack:
                        if hasattr(layer, 'mu'):
                            meansig += layer.sigma.mean().item()
                    for layer in Q.value_linear_relu_stack:
                        if hasattr(layer, 'mu'):
                            meansig += layer.sigma.mean().item()
                    for layer in Q.advantage_linear_relu_stack:
                        if hasattr(layer, 'mu'):
                            meansig += layer.sigma.mean().item()
                else:
                    for layer in Q.linear_relu_stack:
                        if hasattr(layer, 'mu'):
                            meansig += layer.sigma.mean().item()
                print(f"avg sigma: {layer.sigma.mean().item()}")
            print('-----------------------------------')        
        i += 1 # update episode number

    # calculate final average reward
    print('calculating the average reward with the final Q network')
    if external_testing == False:
        final_avgreward = calc_performance(env,device,Q,None,final_performance_sampleN)
        avgperformances.append(final_avgreward)
        print(f'final average reward: {final_avgreward}')
    else:
        torch.save(Q, f"{testwd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN_episode{i}.pt")

    # save results and performance metrics.
    ## save last model and the best model (in terms of rewards)
    if env.envID in ['Env1.0','Env1.1']:
        # last model
        wd = './deepQN results'
        torch.save(Q, f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pt")
        # best model
        
        torch.save(Q, f"{wd}/bestQNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pt")

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
    ## save MSE
    if calc_MSE:
        np.save(f"{wd}/MSE_{env.envID}_par{env.parset}_dis{env.discset}_DQN.npy", MSE)
    return Q_discrete, policy, MSE, avgperformances, final_avgreward



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
        self.index = 0
        self.size = 0
        self.buffer_size = max_size

    def add(self, state, action, reward, next_state, done):
        self.states_buffer[self.index] = state
        self.actions_buffer[self.index] = action
        self.rewards_buffer[self.index] = reward
        self.next_states_buffer[self.index] = next_state
        self.done_buffer[self.index] = done
        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=True)
        states = self.states_buffer[indices]
        actions = self.actions_buffer[indices]
        rewards = self.rewards_buffer[indices]
        next_states = self.next_states_buffer[indices]
        done = self.done_buffer[indices]
        return states, actions, rewards, next_states, done
    
def pretrain(env, nq, memory, batch_size, PrioritizedReplay, max_priority):
    # Make a bunch of random actions from a random state and store the experiences
    reset = True
    memadd = 0 # number of transitions added to memory
    n = nq.n
    while memadd < batch_size:
        if reset == True:
            if env.envID in ['Env1.0', 'Env1.1']:
                env.reset([-1,-1,-1,-1,-1,-1])
                state = env.state
                reset = False
        # Make a random action
        action = np.random.randint(0, env.actionspace_dim[0])
        reward, done, _ = env.step(action)
        next_state = env.state
        if done:
            nq.add(state, action, reward, next_state, done, memory, PrioritizedReplay)
            reset = True
            memadd += n
        else:
            # increase memadd by 1 if nq is full
            if len(nq.queue) == n-1:
                memadd += 1
            nq.add(state, action, reward, next_state, done, memory, PrioritizedReplay)
            state = next_state
    nq.queue = [] # clear the n-step queue
    nq.rqueue = [] 
    

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

def test_model(Q, reachable_states, reachable_actions, Qopt, noisy, device):
    """
    If there is a optimal Q calculate (perhaps from value iteration) MSE loss compared to the optimal Q.
    """
    Q.eval()
    if noisy: # disable noise when evaluating
        Q.disable_noise()
        with torch.no_grad():
            testloss = compute_loss(Q, reachable_states, reachable_actions, Qopt).item()
        Q.enable_noise()
    else:
        with torch.no_grad():
            testloss = compute_loss(Q, reachable_states, reachable_actions, Qopt).item()
    return testloss
    #print(f"Test Error Avg loss: {test_loss:>8f}\n")
