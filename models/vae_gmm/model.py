"""
Core GMM-VAE model logic shared across all image sizes.
This module contains the main model class and shared functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np


class GMMVAE(nn.Module):
    """GMM-VAE model with learnable GMM parameters."""

    def __init__(self, encoder, decoder, latent_dim=10, num_clusters=10):
        """
        Initialize GMM-VAE model.

        Args:
            encoder: Encoder network instance
            decoder: Decoder network instance
            latent_dim: Dimension of latent space
            num_clusters: Number of GMM clusters
        """
        super(GMMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

        self.encoder = encoder
        self.decoder = decoder
        self.gmm_means = nn.Parameter(torch.randn(num_clusters, latent_dim))
        self.gmm_logvars = nn.Parameter(torch.randn(num_clusters, latent_dim))

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def gmm_loss(self, z):
        """Compute GMM loss with hard assignment."""
        distances = torch.cdist(z, self.gmm_means)
        cluster_assignments = torch.argmin(distances, dim=1)
        chosen_means = self.gmm_means[cluster_assignments]
        chosen_logvars = self.gmm_logvars[cluster_assignments]
        log_probs = -0.5 * (
            torch.sum((z - chosen_means) ** 2 / torch.exp(chosen_logvars), dim=-1) +
            torch.sum(chosen_logvars, dim=-1)
        )
        return -log_probs.mean()

    def compute_loss(self, x, recon, mu, logvar, z, loss_weights):
        """
        Compute total loss.

        Args:
            x: Input images
            recon: Reconstructed images
            mu: Latent mean
            logvar: Latent log variance
            z: Sampled latent code
            loss_weights: Dictionary with keys 'reconstruction', 'kl', 'gmm'

        Returns:
            Dictionary with individual losses and total loss
        """
        # Reconstruction loss (BCE)
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # GMM loss
        gmm_loss_val = self.gmm_loss(z)

        # Total loss
        total_loss = (
            loss_weights['reconstruction'] * recon_loss +
            loss_weights['kl'] * kl_loss +
            loss_weights['gmm'] * gmm_loss_val
        )

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'gmm_loss': gmm_loss_val.item()
        }

    def extract_latent_features(self, data_loader, device):
        """Extract latent features for all data."""
        self.eval()
        z_list, labels_list = [], []

        with torch.no_grad():
            for batch, target in data_loader:
                batch = batch.to(device)
                mu, _ = self.encode(batch)
                z_list.append(mu.cpu().numpy())
                labels_list.append(target.numpy())

        z_data = np.vstack(z_list)
        labels = np.hstack(labels_list)

        # Sort lexicographically to ensure deterministic ordering
        # (GMM k-means initialization depends on sample order in the array)
        sort_idx = np.lexsort(z_data.T[::-1])
        return z_data[sort_idx], labels[sort_idx]

    def fit_gmm_and_evaluate(self, z_data, labels, covariance_type='diag', n_init=10):
        """
        Fit GMM on latent space and compute clustering metrics.

        Returns:
            Dictionary with gmm model and predicted labels
        """
        gmm = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=0,
            reg_covar=1e-4  # Increased regularization for numerical stability
        )
        gmm_labels = gmm.fit_predict(z_data)

        return {
            'gmm': gmm,
            'gmm_labels': gmm_labels
        }
