"""
Diffusion VAE-GMM model: combines a VAE-GMM with a latent-space denoiser.

Two-stage model:
  Stage 1: Train VAE-GMM (encoder, decoder, GMM prior) end-to-end
  Stage 2: Freeze VAE, train denoiser on latent codes using DDPM framework

The denoiser operates purely in latent space (MLP, not CNN), so it is
dataset-agnostic. Only the encoder/decoder are size-specific.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np

from .denoiser_network import LatentDenoiser
from .diffusion_utils import get_beta_schedule, q_sample, extract


class DiffusionGMMVAE(nn.Module):
    """GMM-VAE model with latent-space diffusion denoiser."""

    def __init__(self, encoder, decoder, latent_dim=10, num_clusters=10,
                 diffusion_config=None):
        """
        Args:
            encoder: Encoder network instance
            decoder: Decoder network instance
            latent_dim: Dimension of latent space
            num_clusters: Number of GMM clusters
            diffusion_config: Dict with denoiser/diffusion hyperparameters
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

        # VAE components (same as GMMVAE)
        self.encoder = encoder
        self.decoder = decoder
        self.gmm_means = nn.Parameter(torch.randn(num_clusters, latent_dim))
        self.gmm_logvars = nn.Parameter(torch.randn(num_clusters, latent_dim))

        # Diffusion config
        if diffusion_config is None:
            diffusion_config = {}
        self.diffusion_config = diffusion_config

        # Denoiser
        self.denoiser = LatentDenoiser(
            latent_dim=latent_dim,
            time_embed_dim=diffusion_config.get('time_embed_dim', 32),
            hidden_dim=diffusion_config.get('hidden_dim', 128),
            num_layers=diffusion_config.get('num_layers', 3),
        )

        # Diffusion schedule (registered as buffers for device management)
        timesteps = diffusion_config.get('timesteps', 100)
        schedule = diffusion_config.get('schedule', 'linear')
        beta_start = diffusion_config.get('beta_start', 0.0001)
        beta_end = diffusion_config.get('beta_end', 0.02)

        betas, alphas, alphas_cumprod, alphas_cumprod_prev = get_beta_schedule(
            schedule, timesteps, beta_start, beta_end
        )
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

    # ─── VAE methods (from models/vae_gmm/model.py) ────────────────────

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
        """Compute VAE total loss (Phase 1 training)."""
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        gmm_loss_val = self.gmm_loss(z)

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

        # Sort lexicographically for deterministic GMM initialization
        sort_idx = np.lexsort(z_data.T[::-1])
        return z_data[sort_idx], labels[sort_idx]

    def fit_gmm_and_evaluate(self, z_data, labels, covariance_type='diag', n_init=10):
        """Fit GMM on latent space and compute clustering metrics."""
        gmm = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=0,
            reg_covar=1e-4
        )
        gmm_labels = gmm.fit_predict(z_data)
        return {
            'gmm': gmm,
            'gmm_labels': gmm_labels
        }

    # ─── Diffusion methods ──────────────────────────────────────────────

    def freeze_vae(self):
        """Freeze all VAE parameters (for Phase 2 denoiser training)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.gmm_means.requires_grad = False
        self.gmm_logvars.requires_grad = False

    def unfreeze_vae(self):
        """Unfreeze all VAE parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        self.gmm_means.requires_grad = True
        self.gmm_logvars.requires_grad = True

    def compute_denoiser_loss(self, z_0, max_timestep):
        """
        Compute denoiser training loss (MSE on noise prediction).

        Args:
            z_0: Clean latent vectors (batch_size, latent_dim)
            max_timestep: Maximum timestep to sample from

        Returns:
            MSE loss between predicted and actual noise
        """
        batch_size = z_0.shape[0]
        device = z_0.device

        t = torch.randint(0, max_timestep, (batch_size,), device=device).long()
        eps = torch.randn_like(z_0)
        z_t = q_sample(z_0, t, eps, self.alphas_cumprod)

        eps_pred = self.denoiser(z_t, t)
        loss = F.mse_loss(eps_pred, eps)
        return loss

    @torch.no_grad()
    def onestep_denoise(self, z, eval_t, n_mc=50):
        """
        One-step x0 prediction with Monte Carlo averaging.

        Args:
            z: Clean latent vectors (batch_size, latent_dim)
            eval_t: Timestep to noise/denoise at
            n_mc: Number of MC samples to average

        Returns:
            Denoised latent vectors (batch_size, latent_dim)
        """
        B = z.shape[0]
        z_sum = torch.zeros_like(z)

        for _ in range(n_mc):
            t = torch.full((B,), eval_t, device=z.device, dtype=torch.long)
            eps = torch.randn_like(z)
            z_t = q_sample(z, t, eps, self.alphas_cumprod)
            eps_pred = self.denoiser(z_t, t)
            sqrt_ab = extract(torch.sqrt(self.alphas_cumprod), t, z.shape)
            sqrt_1_ab = extract(torch.sqrt(1.0 - self.alphas_cumprod), t, z.shape)
            z_den = (z_t - sqrt_1_ab * eps_pred) / sqrt_ab.clamp(min=1e-5)
            z_sum += z_den

        return z_sum / n_mc

    def extract_denoised_latent_features(self, data_loader, device, denoise_t, n_mc=50):
        """
        Extract latent features and apply one-step denoising.

        Args:
            data_loader: DataLoader for the dataset
            device: Device to run on
            denoise_t: Timestep for one-step denoising
            n_mc: Number of MC samples

        Returns:
            (z_denoised, labels) - both sorted lexicographically
        """
        self.eval()
        z_list, labels_list = [], []

        with torch.no_grad():
            for batch, target in data_loader:
                batch = batch.to(device)
                mu, _ = self.encode(batch)
                z_list.append(mu)
                labels_list.append(target)

        z_all = torch.cat(z_list, 0)
        labels = torch.cat(labels_list, 0).cpu().numpy()

        # Denoise
        z_denoised = self.onestep_denoise(z_all, denoise_t, n_mc)
        z_denoised_np = z_denoised.cpu().numpy()

        # Sort lexicographically
        sort_idx = np.lexsort(z_denoised_np.T[::-1])
        return z_denoised_np[sort_idx], labels[sort_idx]
