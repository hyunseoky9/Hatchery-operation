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

def DQN(env,num_episodes,epdecayopt,DDQN,DuelingDQN,PrioritizedReplay):
    # train using Deep Q Network
    # env: environment class object
    # num_episodes: number of episodes to train
    # epdecayopt: epsilon decay option
    
    # parameters
    ## NN parameters
    # DQN
    state_size = len(env.statespace_dim)
    action_size = env.actionspace_dim[0]
    hidden_size = 30
    hidden_num = 3
    # Dueling DQN
    hidden_num_shared = 1
    hidden_num_split = 1
    hidden_size_shared = 30
    hidden_size_split = 30
    # Prioritized Replay
    alpha = 0.6 # priority importance
    beta0 = 0.2 # initial beta
    per_epsilon = 1e-6 # small value to avoid zero priority
    max_abstd = 1 # initial max priority
    ## memory parameters
    memory_size = 1000 # memory capacity
    batch_size = 100 # experience mini-batch size
    ## etc.
    lr = 0.01 # starting learning rate
    min_lr = 0.00001  # Set the minimum learning rate
    gamma = env.gamma
    max_steps = 1000 # max steps per episode
    ## cycles
    training_cycle = 7 # number of steps where the network is trained
    target_update_cycle = 10 # number of steps where the target network is updated
    ## normalization parameters
    state_max = torch.tensor(env.statespace_dim, dtype=torch.float32) - 1
    state_min = torch.zeros([len(env.statespace_dim)], dtype=torch.float32)

    # initialization
    ## print out extension feature usage
    print(f'DuelingDQN: {DuelingDQN}\n DDQN: {DDQN}\n PrioritizedReplay: {PrioritizedReplay}')
    if PrioritizedReplay:
        print(f'alpha: {alpha}, beta0: {beta0}, per_epsilon: {per_epsilon}')

    ## initialize NN
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    if DuelingDQN:
        Q = DuelQNN(state_size, action_size, hidden_size_shared, hidden_size_split, hidden_num_shared, hidden_num_split, lr, state_min, state_max).to(device)
        Q_target = DuelQNN(state_size, action_size, hidden_size_shared, hidden_size_split, hidden_num_shared, hidden_num_split, lr, state_min, state_max).to(device)
    else:
        Q = QNN(state_size, action_size, hidden_size, hidden_num, lr, state_min, state_max).to(device)
        Q_target = QNN(state_size, action_size, hidden_size, hidden_num, lr, state_min, state_max).to(device)
    Q_target.load_state_dict(Q.state_dict())  # Copy weights from Q to Q_target
    Q_target.eval()  # Set target network to evaluation mode (no gradient updates)

    ## initialize memory
    if PrioritizedReplay:
        memory = PMemory(memory_size, alpha, per_epsilon, max_abstd)
        beta = beta0
        pretrain(env,memory,batch_size,PrioritizedReplay,memory.max_abstd) # prepopulate memory
    else:
        memory = Memory(memory_size, state_size, len(env.actionspace_dim))
        pretrain(env,memory,batch_size,PrioritizedReplay,0) # prepopulate memory
    print(f'Pretraining memory with {memory_size} experiences')


    ## state initialization setting
    if env.envID == 'Env1.0':
        initlist = [-1,-1,-1,-1,-1,-1] # all random
        reachables = env.reachable_state_actions()
        reachable_states = torch.tensor([env._unflatten(i[0]) for i in reachables], dtype=torch.float32)
        reachable_uniquestateid = torch.tensor(env.reachable_states(), dtype=torch.int64)
        reachable_actions = torch.tensor([i[1] for i in reachables], dtype=torch.int64).unsqueeze(1)
        
    # load Q function from the value iteration for calculating MSE
    if env.envID == 'Env1.0':
        with open(f"value iter results/Q_Env1.0_par{env.parset}_dis{env.discset}_valiter.pkl", "rb") as file:
            Q_vi = pickle.load(file)
        Q_vi = torch.tensor(Q_vi[reachable_uniquestateid].flatten(), dtype=torch.float32)
    MSE = []
    
    # initialize counters
    j = 0 # training cycle counter
    i = 0 # peisode num
    # run through the episodes
    while i < num_episodes: #delta > theta:
        # update epsilon
        ep = epsilon_update(i,epdecayopt,num_episodes) 
        # initialize state that doesn't start from terminal
        env.reset(initlist) # random initialization
        S = env.state
        done = False

        t = 0 # timestep num
        while done == False:    
            if t > 0:    
                a = choose_action(S, Q, ep, action_size)
            else:
                a = random.randint(0, action_size-1) # first action in the episode is random for added exploration
            reward, done, rate = env.step(a) # take a step
            if PrioritizedReplay:
                memory.add(memory.max_abstd, (S, a, reward, env.state, done)) # add experience to memory
            else:
                memory.add(S, a, reward, env.state, done) # add experience to memory
            S = env.state # update state
            if t >= max_steps: # finish episode if max steps reached even if terminal state not reached
                done = True
            # train network
            if j % training_cycle == 0:
                # Sample mini-batch from memory
                if PrioritizedReplay:
                    mini_batch, idxs, weights = memory.sample(batch_size, beta)
                    states, actions, rewards, next_states, dones = zip(*mini_batch)
                    dones = np.array(dones)
                    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                else:
                    states, actions, rewards, next_states, dones = memory.sample(batch_size)
                    weights = np.ones(batch_size)
                    actions = torch.tensor(actions, dtype=torch.int64)
                states = torch.tensor(states, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                weights = torch.tensor(weights, dtype=torch.float32).to(device)
                # Train network
                # Set target_Qs to 0 for states where episode ends
                episode_ends = np.where(dones == True)[0]
                target_Qs = Q_target(next_states)
                target_Qs[episode_ends] = torch.zeros(action_size)
                if DDQN:
                    next_actions = torch.argmax(Q(next_states), dim=1)
                    targets = rewards + gamma * target_Qs.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    targets = rewards + gamma * torch.max(target_Qs, dim=1)[0]
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
        if i % 1000 == 0:
            current_lr = Q.optimizer.param_groups[0]['lr']
            print(f"Episode {i}, Learning Rate: {current_lr}")

        if i % 100 == 0:
            mse_value = test_model(Q, reachable_states, reachable_actions, Q_vi, device)
            MSE.append(mse_value)
        
        if PrioritizedReplay:
            beta += (1.0 - beta0)/num_episodes
        Q.scheduler.step() # Decay the learning rate
        #if Q.optimizer.param_groups[0]['lr'] < min_lr:
        #    Q.optimizer.param_groups[0]['lr'] = min_lr
        i += 1 # update episode number

    # save results and performance metrics.
    ## save model
    if env.envID == 'Env1.0':
        wd = './deepQN results'
        torch.save(Q.state_dict(), f"{wd}/QNetwork_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pt")
    ## make a discrete Q table if the environment is discrete and save it
    if env.envID == 'Env1.0':
        Q_discrete = _make_discrete_Q(Q,env,device)
        policy = _get_policy(env,Q_discrete)
        wd = './deepQN results'
        with open(f"{wd}/Q_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pkl", "wb") as file:
            pickle.dump(Q_discrete, file)
        with open(f"{wd}/policy_{env.envID}_par{env.parset}_dis{env.discset}_DQN.pkl", "wb") as file:
            pickle.dump(policy, file)
    ## save MSE
    np.save(f"{wd}/MSE_{env.envID}_par{env.parset}_dis{env.discset}_DQN.npy", MSE)
    return Q_discrete, policy, MSE

def _make_discrete_Q(Q,env,device):
    # make a discrete Q table
    if env.envID == 'Env1.0':
        states = torch.tensor([env._unflatten(i) for i in range(np.prod(env.statespace_dim))], dtype=torch.float32)
        Q_discrete = np.zeros((np.prod(env.statespace_dim),len(env.actions["a"])))
        for i in range(np.prod(env.statespace_dim)):
            Q_discrete[i,:] = Q(states[i].unsqueeze(0)).detach().cpu().numpy()
    return Q_discrete

def choose_action(state, Q, epsilon, action_size):
    # Choose an action
    if random.random() < epsilon:
        action = random.randint(0, action_size-1)
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        Qs = Q(state)
        action = torch.argmax(Qs).item()
    return action

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
    
def pretrain(env, memory, batch_size, PrioritizedReplay, max_priority):
    # Make a bunch of random actions from a random state and store the experiences
    reset = True
    for ii in range(batch_size):
        if reset == True:
            if env.envID == 'Env1.0':
                env.reset([-1,-1,-1,-1,-1,-1])
                state = env.state
                reset = False
        # Make a random action
        action = np.random.randint(0, env.actionspace_dim[0])
        reward, done, _ = env.step(action)
        next_state = env.state

        if done:
            # Add experience to memory
            if PrioritizedReplay:
                memory.add(max_priority, (state, action, reward, next_state, done))
            else:
                memory.add(state, action, reward, next_state, done)

            reset = True
        else:
            # Add experience to memory
            if PrioritizedReplay:
                memory.add(max_priority, (state, action, reward, next_state, done))
            else:
                memory.add(state, action, reward, next_state, done)
            state = next_state

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
        predictions = predictions.gather(1, actions).squeeze(1) # Get Q-values for the selected actions

        td_errors = targets - predictions
        # compute_loss
        # loss = #Q.loss_fn(predictions, targets) # Compute the loss
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
    selected_q_values = q_values.gather(1, actions).squeeze(1) # Get Q-values for the selected actions
    loss = Q.loss_fn(selected_q_values, targetQs) # Compute the loss
    return loss

def test_model(Q, reachable_states, reachable_actions, Qopt, device):
    """
    If there is a optimal Q calculate (perhaps from value iteration) MSE loss compared to the optimal Q.
    """
    Q.eval()
    with torch.no_grad():
        testloss = compute_loss(Q, reachable_states, reachable_actions, Qopt).item()
    return testloss
    #print(f"Test Error Avg loss: {test_loss:>8f}\n")

def normalize(Q, state):
    """
    min-max normalization for discrete states.
    parmaeters: 
        states (torch.Tensor): Input states
        env (object): Environment object
    """
    # Normalize using broadcasting
    state_norm = (state - Q.state_min) / (Q.state_max - Q.state_min)
    return state_norm
    

