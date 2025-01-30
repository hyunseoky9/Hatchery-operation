import torch
import random
import numpy as np

def choose_action(state, Q, epsilon, action_size, distributional,device, drqn=False, hidden=None, previous_action=None):
    if previous_action is not None:
        state_ = state.copy()
        state_.append(previous_action)
    else:
        state_ = state
    # Choose an action
    if random.random() < epsilon:
        
        action = random.randint(0, action_size-1)
        if drqn == True:
            state_ = torch.tensor(state_, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                _, hidden = Q(state_, training=False, hidden=hidden)
            return action, hidden
        else:
            return action
    else:
        state_ = torch.tensor(state_, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        Q.eval()
        if drqn == False: # DQN
            with torch.no_grad():
                Qs = Q(state_)

            if distributional:
                Q_expected = torch.sum(Qs * Q.z, dim=-1) # sum over atoms for each action
                action = torch.argmax(Q_expected).item()
            else:
                action = torch.argmax(Qs).item()
            return action
        else: # DRQN
            with torch.no_grad():
                Qs, hidden = Q(state_, training=False ,hidden=hidden)

            if distributional:
                Q_expected = torch.sum(Qs * Q.z, dim=-1) # sum over atoms for each action
                action = torch.argmax(Q_expected).item()
            else:
                action = torch.argmax(Qs).item()
            return action, hidden
