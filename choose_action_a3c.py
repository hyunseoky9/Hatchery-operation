import torch
import random
import numpy as np

def choose_action_a3c(state, policy, hidden_state):
    # Choose an action
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy, _, hidden_state = policy(state, hidden_state)
    action = torch.multinomial(policy, 1).item() # sample action
    return action, hidden_state
