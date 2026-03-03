"""
Denoiser network for latent space diffusion.

This module implements a timestep-conditioned MLP that predicts noise
in the latent space for the reverse diffusion process.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.

    Similar to Transformer positional encodings, this provides rich
    representations of timestep information to the denoising network.
    """

    def __init__(self, dim):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: Timesteps (batch_size,)

        Returns:
            Embedded timesteps (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LatentDenoiser(nn.Module):
    """
    MLP-based denoising network for latent space diffusion.

    Takes a noisy latent vector z_t and timestep t, and predicts the noise
    component epsilon that was added to the clean latent z_0.
    """

    def __init__(self, latent_dim=10, time_embed_dim=32, hidden_dim=128, num_layers=3):
        """
        Args:
            latent_dim: Dimensionality of latent space
            time_embed_dim: Dimensionality of timestep embedding
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # Main denoising network
        layers = []
        input_dim = latent_dim + time_embed_dim

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z_t, t):
        """
        Predict noise epsilon from noisy latent z_t at timestep t.

        Args:
            z_t: Noisy latent vectors (batch_size, latent_dim)
            t: Timesteps (batch_size,), integer values in [0, T)

        Returns:
            eps_pred: Predicted noise (batch_size, latent_dim)
        """
        # Embed timestep
        t_emb = self.time_mlp(t.float())

        # Concatenate latent and timestep embedding
        z_combined = torch.cat([z_t, t_emb], dim=-1)

        # Predict noise
        eps_pred = self.network(z_combined)

        return eps_pred
