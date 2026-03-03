"""
Diffusion utilities for DDPM in latent space.

This module provides helper functions for forward and reverse diffusion processes
following the DDPM framework (Denoising Diffusion Probabilistic Models).
"""

import torch
import numpy as np


def get_beta_schedule(schedule, timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Generate beta schedule for diffusion process.

    Args:
        schedule: Type of schedule ('linear' or 'cosine')
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values for each timestep (timesteps,)
        alphas: Alpha values (1 - beta) for each timestep (timesteps,)
        alphas_cumprod: Cumulative product of alphas (timesteps,)
        alphas_cumprod_prev: Shifted cumulative product (timesteps,)
    """
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == 'cosine':
        # Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    return betas, alphas, alphas_cumprod, alphas_cumprod_prev


def extract(arr, timesteps, shape):
    """
    Extract values from arr at specified timesteps and reshape to match target shape.

    Args:
        arr: Array to extract from (T,)
        timesteps: Timestep indices (batch_size,)
        shape: Target shape (batch_size, ...)

    Returns:
        Extracted values reshaped to (batch_size, 1, 1, ...)
    """
    batch_size = timesteps.shape[0]
    out = arr.to(timesteps.device)[timesteps]
    # Reshape to (batch_size, 1, 1, ...) for broadcasting
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


def q_sample(z_0, t, noise, alphas_cumprod):
    """
    Forward diffusion: add noise to z_0 at timestep t.

    q(z_t | z_0) = N(sqrt(alpha_cumprod_t) * z_0, (1 - alpha_cumprod_t) * I)

    Args:
        z_0: Clean latent vectors (batch_size, latent_dim)
        t: Timesteps (batch_size,)
        noise: Gaussian noise (batch_size, latent_dim)
        alphas_cumprod: Cumulative product of alphas (T,)

    Returns:
        z_t: Noisy latent vectors (batch_size, latent_dim)
    """
    sqrt_alphas_cumprod_t = extract(torch.sqrt(alphas_cumprod), t, z_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1.0 - alphas_cumprod), t, z_0.shape)

    z_t = sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return z_t


@torch.no_grad()
def p_sample(denoiser, z_t, t, betas, alphas, alphas_cumprod, alphas_cumprod_prev):
    """
    Reverse diffusion: denoise z_t to get z_{t-1} (single step).

    p(z_{t-1} | z_t) = N(mu_theta(z_t, t), sigma_t^2 * I)

    Args:
        denoiser: Denoising network that predicts noise
        z_t: Noisy latent at timestep t (batch_size, latent_dim)
        t: Current timesteps (batch_size,)
        betas: Beta schedule (T,)
        alphas: Alpha schedule (T,)
        alphas_cumprod: Cumulative product of alphas (T,)
        alphas_cumprod_prev: Shifted cumulative product (T,)

    Returns:
        z_t_minus_1: Denoised latent at timestep t-1 (batch_size, latent_dim)
    """
    # Predict noise
    eps_pred = denoiser(z_t, t)

    # Extract schedule values for current timestep
    beta_t = extract(betas, t, z_t.shape)
    alpha_t = extract(alphas, t, z_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1.0 - alphas_cumprod), t, z_t.shape)
    sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / alphas), t, z_t.shape)

    # Compute mean of p(z_{t-1} | z_t)
    # mu = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1-alpha_cumprod_t) * eps)
    model_mean = sqrt_recip_alphas_t * (z_t - beta_t * eps_pred / sqrt_one_minus_alphas_cumprod_t)

    # Compute variance
    # For t > 0, sigma_t^2 = beta_t
    # For t = 0, sigma_t^2 = 0 (no noise added)
    posterior_variance_t = beta_t

    # Sample from N(mu, sigma^2)
    noise = torch.randn_like(z_t)
    # No noise when t == 0
    nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(z_t.shape) - 1)))

    z_t_minus_1 = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    return z_t_minus_1


@torch.no_grad()
def p_sample_loop(denoiser, z_T, timesteps, betas, alphas, alphas_cumprod, alphas_cumprod_prev, device):
    """
    Full reverse diffusion: denoise from z_T to z_0.

    Args:
        denoiser: Denoising network
        z_T: Initial noisy latent (batch_size, latent_dim)
        timesteps: Number of timesteps
        betas, alphas, alphas_cumprod, alphas_cumprod_prev: Diffusion schedules
        device: Device to run on

    Returns:
        z_0: Denoised latent (batch_size, latent_dim)
    """
    z_t = z_T

    # Reverse iterate from T-1 to 0
    for i in reversed(range(timesteps)):
        t = torch.full((z_t.shape[0],), i, device=device, dtype=torch.long)
        z_t = p_sample(denoiser, z_t, t, betas, alphas, alphas_cumprod, alphas_cumprod_prev)

    return z_t
