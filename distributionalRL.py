import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random

def compute_target_distribution(reward, done, gamma, next_probs, best_action, z, atomn, Vmin, Vmax):
    """
    Compute the target distribution for a batch of transitions.

    Parameters:
        reward (torch.Tensor): Rewards for the batch (shape: [batch_size]).
        done (torch.Tensor): Done flags for the batch (shape: [batch_size]).
        gamma (float): Discount factor.
        next_probs (torch.Tensor): Next state probabilities (shape: [batch_size, action_size, atomn]).
        best_action (torch.Tensor): Best actions (shape: [batch_size, 1]).
        z (torch.Tensor): Atom values (shape: [atomn]).
        atomn (int): Number of atoms.
        Vmin (float): Minimum value for atoms.
        Vmax (float): Maximum value for atoms.

    Returns:
        torch.Tensor: Target probabilities (shape: [batch_size, action_size, atomn]).
    """
    batch_size = reward.shape[0]
    action_size = next_probs.shape[1]

    # Expand reward and done to match shape
    reward = reward.unsqueeze(1).unsqueeze(2).expand(batch_size, action_size, atomn)  # Shape: [batch_size, action_size, atomn]
    done = done.unsqueeze(1).unsqueeze(2).expand(batch_size, action_size, atomn)      # Shape: [batch_size, action_size, atomn]

    # Initialize target_z
    target_z = torch.zeros_like(reward)  # Shape: [batch_size, action_size, atomn]

    # Terminal states
    target_z[done] = reward[done]

    # Non-terminal states
    not_done = ~done
    if not_done.any():
        target_z[not_done] = reward[not_done] + gamma * z * next_probs[not_done].gather(2, best_action[not_done].unsqueeze(2)).squeeze(2)

    # Clip and project back to atom support
    target_z = torch.clamp(target_z, Vmin, Vmax)
    target_probs = project_distribution(target_z, z, atomn)  # Shape: [batch_size, action_size, atomn]

    return target_probs


def project_distribution(target_z, z, atomn):
    """
    Project the target distribution back onto the atom support.

    Parameters:
        target_z (torch.Tensor): Target values (shape: [batch_size, action_size, atomn]).
        z (torch.Tensor): Atom values (shape: [atomn]).
        atomn (int): Number of atoms.

    Returns:
        torch.Tensor: Projected target probabilities (shape: [batch_size, action_size, atomn]).
    """
    delta_z = z[1] - z[0]  # Atom spacing
    b = (target_z - z[0]) / delta_z  # Compute positions in the atom space
    lower = torch.floor(b).long()  # Lower atom indices
    upper = torch.ceil(b).long()  # Upper atom indices

    lower = torch.clamp(lower, 0, atomn - 1)
    upper = torch.clamp(upper, 0, atomn - 1)

    batch_size, action_size, _ = target_z.shape

    # Initialize target_probs
    target_probs = torch.zeros(batch_size, action_size, atomn, device=target_z.device)

    # Distribute probabilities across lower and upper bounds
    target_probs.scatter_add_(2, lower, (upper - b).clamp(0, 1))
    target_probs.scatter_add_(2, upper, (b - lower).clamp(0, 1))

    return target_probs