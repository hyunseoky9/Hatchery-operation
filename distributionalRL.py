import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random

def compute_target_distribution(reward, done, gamma, nstep, next_probs, best_action, z, atomn, Vmin, Vmax):
    """
    Compute the target distribution for a batch of transitions.

    Parameters:
        reward (torch.Tensor): Rewards for the batch (shape: [batch_size]).
        done (np.ndarray or torch.Tensor): Done flags for the batch (shape: [batch_size]).
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

    # Expand reward and done to match shape
    reward = reward.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, atomn)  # Shape: [batch_size, action_size, atomn]
    # Ensure all inputs are tensors
    done = torch.tensor(done, dtype=torch.bool, device=next_probs.device)  # Convert to tensor
    done = done.unsqueeze(1).unsqueeze(2).float()

    # Bellman operation
    target_z = reward + (1 - done) * (gamma**nstep) * z.unsqueeze(0).unsqueeze(0)  # Shape: [batch_size, action_size, atomn]
    # Clip and project back to atom support
    target_z = torch.clamp(target_z, Vmin, Vmax)  # Clip values

    # calculate the next state probability with the best action
    best_action_probs = next_probs.gather(1, best_action.unsqueeze(-1).expand(-1,-1,atomn))  # [batch_size, 1, atomn]

    target_probs = project_distribution(target_z, z, atomn, best_action_probs)  # Shape: [batch_size, action_size, atomn]

    return target_probs


def project_distribution(target_z, z, atomn, probs):
    """
    Project the target distribution back onto the atom support.

    Parameters:
        target_z (torch.Tensor): Target values (shape: [batch_size, 1, atomn]).
        z (torch.Tensor): Atom values (shape: [atomn]).
        atomn (int): Number of atoms.
        probs (torch.Tensor): Probabilities for the target_z (shape: [batch_size, 1, atomn]).

    Returns:
        torch.Tensor: Projected target probabilities (shape: [batch_size, action_size, atomn]).
    """
    delta_z = z[1] - z[0]  # Atom spacing
    b = torch.abs((target_z - z[0]) / delta_z)  # Compute absolute positions in the atom space
    b = b.clamp(0, atomn - 1) # ensure b is within bounds
    lower = torch.floor(b).long()  # Lower atom indices
    upper = torch.ceil(b).long()  # Upper atom indices
    
    lower = torch.clamp(lower, 0, atomn - 1) # ensure lower index is within bounds
    upper = torch.clamp(upper, 0, atomn - 1) # ensure upper index is withint bounds

    # Identify exact matches (where b is an integer)
    exact_match = (b % 1 == 0)

    batch_size, action_size, _ = target_z.shape

    # Initialize target_probs
    target_probs = torch.zeros(batch_size, action_size, atomn, device=target_z.device)

    # Distribute probabilities across lower and upper bounds
    target_probs.scatter_add_(2, lower, exact_match.float()*probs)
    target_probs.scatter_add_(2, lower, probs*(upper - b))
    target_probs.scatter_add_(2, upper, probs*(b - lower))

    return target_probs