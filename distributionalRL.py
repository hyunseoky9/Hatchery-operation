import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random

def compute_target_distribution(reward, done, gamma, next_probs, best_action, z, atomn, Vmin, Vmax):
    # Compute target distribution
    if done:
        target_z = reward.unsqueeze(1).expand(-1, atomn)  # Only the immediate reward
    else:
        # Compute distributional Bellman update
        target_z = reward.unsqueeze(1) + gamma * z * next_probs.gather(1, best_action)

    # Clip and project back to atom support
    target_z = torch.clamp(target_z, Vmin, Vmax)
    target_probs = project_distribution(target_z, z, atomn)  # Implement projection step
    return target_probs

def project_distribution(target_z, z, atomn):
    delta_z = z[1] - z[0]  # Atom spacing
    b = (target_z - z[0]) / delta_z  # Compute positions of target in the atom space
    lower = torch.floor(b).long()  # Lower atom indices
    upper = torch.ceil(b).long()  # Upper atom indices

    lower = torch.clamp(lower, 0, atomn - 1)
    upper = torch.clamp(upper, 0, atomn - 1)

    target_probs = torch.zeros_like(z).unsqueeze(0).repeat(target_z.size(0), 1)
    target_probs.scatter_add_(1, lower, (upper - b).clamp(0, 1))
    target_probs.scatter_add_(1, upper, (b - lower).clamp(0, 1))

    return target_probs