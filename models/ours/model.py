"""
Core Ours (Clustering with Latent Atlas Selection & Training) model.
StaticGMMVAE with learnable GMM prior and GMM overwriting capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np


class StaticGMMVAE(nn.Module):
    """
    VAE with learnable GMM prior.

    This model uses a Gaussian Mixture Model (GMM) as the prior distribution
    instead of the standard normal distribution. The GMM parameters are trainable
    and can be overwritten with refit GMM statistics during training.

    Key features:
    - Trainable GMM means, log-variances, and mixture weights
    - Logsumexp-based GMM loss for stable gradients
    - GMM overwriting capability for refined cluster structure
    """

    def __init__(self, encoder, decoder, latent_dim, num_clusters):
        """
        Initialize StaticGMMVAE.

        Args:
            encoder: Encoder network instance
            decoder: Decoder network instance
            latent_dim: Dimension of latent space
            num_clusters: Number of GMM clusters
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

        # Trainable GMM parameters
        self.gmm_means = nn.Parameter(torch.randn(num_clusters, latent_dim) * 0.05)
        self.gmm_logvars = nn.Parameter(torch.zeros(num_clusters, latent_dim))
        self.logit_pi = nn.Parameter(torch.zeros(num_clusters))

        self.K, self.D = num_clusters, latent_dim

    def encode(self, x):
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            mu: Mean of latent distribution [B, D]
            logvar: Log-variance of latent distribution [B, D]
        """
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE sampling.

        Args:
            mu: Mean [B, D]
            logvar: Log-variance [B, D]

        Returns:
            z: Sampled latent code [B, D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent code to reconstruction.

        Args:
            z: Latent code [B, D]

        Returns:
            recon: Reconstructed image [B, C, H, W]
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            recon: Reconstructed image [B, C, H, W]
            mu: Latent mean [B, D]
            logvar: Latent log-variance [B, D]
            z: Sampled latent code [B, D]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def gmm_loss(self, z, eps=1e-6):
        """
        Compute GMM prior loss using logsumexp for numerical stability.

        This loss encourages latent codes to cluster around GMM components.
        Uses log-space computation to prevent numerical underflow.

        Args:
            z: Latent codes [B, D]
            eps: Small constant for numerical stability

        Returns:
            loss: Negative log-likelihood under GMM prior (scalar)
        """
        # Compute log-likelihood for each GMM component
        # diff: [B, K, D]
        diff = z[:, None, :] - self.gmm_means[None, :, :]

        # var: [1, K, D]
        var = torch.exp(self.gmm_logvars)[None, :, :] + eps

        # logp: [B, K] - log-likelihood for each component
        logp = -0.5 * torch.sum(diff * diff / var + torch.log(var), dim=-1)

        # log_pi: [1, K] - log mixture weights
        log_pi = F.log_softmax(self.logit_pi, dim=0)[None, :]

        # Logsumexp over components: [B]
        log_likelihood = torch.logsumexp(log_pi + logp, dim=1)

        return -log_likelihood.mean()

    def pi_balance_loss(self):
        """
        Optional loss to encourage balanced cluster assignments.

        Returns:
            loss: MSE between current mixture weights and uniform distribution
        """
        pi = F.softmax(self.logit_pi, dim=0)
        uniform = torch.full_like(pi, 1.0 / pi.numel())
        return F.mse_loss(pi, uniform)

    def compute_loss(self, x, recon, mu, logvar, z, loss_weights):
        """
        Compute total VAE loss with GMM prior.

        Args:
            x: Input images [B, C, H, W]
            recon: Reconstructed images [B, C, H, W]
            mu: Latent mean [B, D]
            logvar: Latent log-variance [B, D]
            z: Sampled latent code [B, D]
            loss_weights: Dictionary with keys 'reconstruction', 'kl', 'gmm'

        Returns:
            Dictionary with individual losses and total loss
        """
        # Reconstruction loss (BCE by default, MSE if configured)
        recon_type = loss_weights.get('reconstruction_type', 'bce')
        if recon_type == 'mse':
            recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        else:
            recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)

        # KL divergence with standard normal
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # GMM prior loss
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
        """
        Extract latent features for all data in loader.

        Args:
            data_loader: DataLoader instance
            device: torch.device

        Returns:
            z_data: Latent features [N, D] (numpy array)
            labels: Ground truth labels [N] (numpy array)
        """
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

    def fit_gmm_and_evaluate(self, z_data, labels, covariance_type='diag'):
        """
        Fit GMM on latent space.

        Args:
            z_data: Latent features [N, D] (numpy array)
            labels: Ground truth labels [N] (numpy array)
            covariance_type: GMM covariance type ('diag', 'full', 'spherical')

        Returns:
            Dictionary with gmm model and predicted labels
        """
        gmm = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type=covariance_type,
            random_state=0,
            reg_covar=1e-4
        )
        gmm_labels = gmm.fit_predict(z_data)

        return {
            'gmm': gmm,
            'gmm_labels': gmm_labels
        }

    def overwrite_gmm_from_sklearn(self, gmm, ema_decay=None):
        """
        Overwrite trainable GMM parameters with sklearn GMM statistics.

        This method is typically called after fitting sklearn GMM on embeddings
        to refine the cluster structure. Can optionally use EMA for smooth updates.

        Args:
            gmm: Fitted sklearn.mixture.GaussianMixture instance
            ema_decay: Optional EMA decay factor (0.9 recommended). If None, direct copy.
        """
        device = self.gmm_means.device
        K, D = self.K, self.D

        # Extract means
        means = torch.from_numpy(gmm.means_).float().to(device)

        # Extract covariances (handle different types)
        if gmm.covariance_type == "full":
            # Extract diagonal from full covariance matrices
            var = np.stack([np.diag(c) for c in gmm.covariances_], axis=0)
        elif gmm.covariance_type == "diag":
            var = gmm.covariances_
        elif gmm.covariance_type == "spherical":
            var = np.repeat(gmm.covariances_[:, None], D, axis=1)
        else:
            raise ValueError(f"Unsupported covariance_type: {gmm.covariance_type}")

        logvars = torch.from_numpy(np.log(var + 1e-6)).float().to(device)

        # Extract mixture weights
        logit_pi = torch.from_numpy(np.log(gmm.weights_ + 1e-10)).float().to(device)

        # Update parameters (with optional EMA)
        if ema_decay is None:
            # Direct copy
            self.gmm_means.data.copy_(means)
            self.gmm_logvars.data.copy_(logvars)
            self.logit_pi.data.copy_(logit_pi)
        else:
            # EMA update
            a = float(ema_decay)
            self.gmm_means.data.copy_((1 - a) * self.gmm_means.data + a * means)
            self.gmm_logvars.data.copy_((1 - a) * self.gmm_logvars.data + a * logvars)
            self.logit_pi.data.copy_((1 - a) * self.logit_pi.data + a * logit_pi)
