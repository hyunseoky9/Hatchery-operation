import torch
import random
import numpy as np

def choose_action(state, Q, epsilon, action_size, distributional,device):
    # Choose an action
    if random.random() < epsilon:
        action = random.randint(0, action_size-1)
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        Qs = Q(state)

        if distributional:
            Q_expected = torch.sum(Qs * Q.z, dim=-1) # sum over atoms for each action
            action = torch.argmax(Q_expected).item()
        else:
            action = torch.argmax(Qs).item()
    return action
