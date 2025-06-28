import torch
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F


def generate_gaussian_mixture(N, M, means, stds=None, weights=None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    Generate N data points from a 1D Gaussian Mixture distribution with M components.

    Args:
        N (int): Number of data points to generate.
        M (int): Number of Gaussian components.
        means (list or torch.Tensor): Means of the Gaussian components (length M).
        stds (list or torch.Tensor): Standard deviations of the Gaussian components (length M).
        weights (list or torch.Tensor, optional): Weights of the Gaussian components (length M).
                                                  If None, weights are uniform.

    Returns:
        torch.Tensor: Generated data points of shape (N,).
        function: A function handle to evaluate the density of the Gaussian mixture.
    """
    # Ensure means, stds, and weights are tensors
    means = torch.tensor(means, dtype=torch.float32)

    if stds is None:
        stds = torch.ones(M, dtype=torch.float32)
    else:
        stds = torch.tensor(stds, dtype=torch.float32)

    if weights is None:
        weights = torch.ones(M, dtype=torch.float32) / M
    else:
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize weights

    means, stds, weights = means.to(device), stds.to(device), weights.to(device)
    # Check validity of inputs
    assert len(means) == M, "Means length must equal M."
    assert len(stds) == M, "Stds length must equal M."
    assert len(weights) == M, "Weights length must equal M."

    # Sample component indices based on weights
    component_indices = torch.multinomial(weights, N, replacement=True)

    # Sample data points from the selected components
    data = torch.normal(mean=means[component_indices], std=stds[component_indices])


    return data


def log_density_gmm(x, means, stds):
    """
    Args:
        x (torch.Tensor): A tensor of shape (N,), representing data points.
        means (list or torch.Tensor): Means of the Gaussian components (length M).
        stds (list or torch.Tensor): Standard deviations of the Gaussian components (length M).
    Returns:
        log_density (torch.Tensor): A tensor of shape (N,), representing the log density for each x.
    """
    # Ensure x is a 1D tensor
    if x.ndim == 1:
        x = x.unsqueeze(-1)  # Convert to shape (N, 1) for broadcasting
    elif x.ndim == 2 and x.shape[1] != 1:
        raise ValueError("Input x must be a 1D tensor or a 2D tensor with shape (N, 1).")

    means = torch.tensor(means).to(x)  # Ensure means are torch.Tensor and on the same device
    stds = torch.tensor(stds).to(x)  # Ensure stds are torch.Tensor and on the same device

    # Calculate Gaussian densities
    exponent = -0.5 * ((x - means) / stds) ** 2  # Shape: (N, M)
    normalizer = 1 / (stds * torch.sqrt(torch.tensor(2 * torch.pi)))
    densities = normalizer * torch.exp(exponent)  # Shape: (N, M)

    # Log-sum-exp trick for numerical stability
    log_densities = torch.log(densities.sum(dim=1))  # Sum over M components for each x

    return log_densities




if __name__ == '__main__':
    N = 10000
    M = 2
    cond_mean, cond_std = [0.5, 1.5], [0.25, 0.25]
    uncond_mean, uncond_std = [-1, 1], [0.5, 0.5]

    data = generate_gaussian_mixture(N, M, cond_mean, cond_std)
    plt.figure()
    plt.hist(data.numpy(), bins=100, density=True)
    plt.title('conditional')
    plt.xlim(-3,3)

    data = generate_gaussian_mixture(N, M, uncond_mean, uncond_std)
    plt.figure()
    plt.hist(data.numpy(), bins=100, density=True)
    plt.title('unconditional')
    plt.xlim(-3, 3)
    plt.show()