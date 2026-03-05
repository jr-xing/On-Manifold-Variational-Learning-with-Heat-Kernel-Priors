"""
Model-agnostic atlas visualization utilities.

This module provides wrappers around models.vae_clast.atlas_utils functions
to make them compatible with any VAE-GMM model variant.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import core functions from CLAST atlas utilities
from models.ours.atlas_utils import (
    sharpen_probs,
    ensure_dir,
    compute_cluster_purity,
    save_cluster_sample_grids,
    build_cluster_atlas_A,
    save_atlas_A_images,
    build_cluster_atlas_laplacian,
    save_atlas_laplacian_images,
    build_cluster_atlas_geodesic_medoid,
    save_atlas_geodesic_images,
    save_refit_gmm_means_atlas,
    save_vae_prior_means_atlas
)


@torch.no_grad()
def compute_gmm_probabilities_generic(model, z_data_torch, device):
    """
    Compute GMM cluster probabilities for any VAE-GMM model.

    Works with:
    - StaticGMMVAE (has logit_pi for mixture weights)
    - GMMVAE (no mixture weights, assumes uniform)
    - Other VAE-GMM variants
    - Baseline GMM/KMeans (uses sklearn model's predict_proba)

    Args:
        model: VAE model with gmm_means, gmm_logvars attributes, or baseline model with gmm/kmeans attribute
        z_data_torch: Latent codes [N, D] on device
        device: torch.device

    Returns:
        probs: Cluster probabilities [N, K] (numpy array)
    """
    model.eval()

    # Check if this is a baseline model (has gmm or kmeans attribute from sklearn)
    if hasattr(model, 'gmm') and model.gmm is not None:
        # Baseline GMM model - use sklearn's predict_proba
        z_data_np = z_data_torch.cpu().numpy()
        probs = model.gmm.predict_proba(z_data_np)
        return probs
    elif hasattr(model, 'kmeans') and model.kmeans is not None:
        # Baseline KMeans model - create one-hot probabilities
        z_data_np = z_data_torch.cpu().numpy()
        labels = model.kmeans.predict(z_data_np)
        probs = np.zeros((len(labels), model.num_clusters))
        probs[np.arange(len(labels)), labels] = 1.0
        return probs

    # Standard VAE-GMM model with PyTorch gmm_means/gmm_logvars
    # Compute log-likelihoods per component
    # diff: [N, K, D]
    diff = z_data_torch[:, None, :] - model.gmm_means[None, :, :]
    # var: [1, K, D]
    var = torch.exp(model.gmm_logvars)[None, :, :] + 1e-6
    # logp: [N, K] - log p(z|k) for each component
    logp = -0.5 * torch.sum(diff * diff / var + torch.log(var), dim=-1)

    # Check if model has mixture weights (logit_pi)
    if hasattr(model, 'logit_pi'):
        # Use trained mixture weights
        log_pi = F.log_softmax(model.logit_pi, dim=0)[None, :]  # [1, K]
    else:
        # Assume uniform mixture weights
        K = model.num_clusters
        log_pi = torch.full((1, K), -np.log(K), device=device)

    # Combine: log p(k|z) ∝ log p(z|k) + log p(k)
    log_prob = log_pi + logp
    probs = torch.softmax(log_prob, dim=1)

    return probs.cpu().numpy()


@torch.no_grad()
def collect_all_images(data_loader, device):
    """
    Collect all images from data loader into single tensor.

    Args:
        data_loader: PyTorch data loader
        device: torch.device

    Returns:
        X_all: All images [N, C, H, W] on CPU
    """
    images_list = []
    for batch_data, _ in data_loader:
        images_list.append(batch_data)

    X_all = torch.cat(images_list, dim=0).cpu()
    return X_all


def save_cluster_sample_grids_generic(X_all, Y_all, probs, conf_thresh, out_dir,
                                     epoch, class_names, sample_topn=32, nrow=8):
    """
    Save grid of top-confidence samples per cluster.

    Model-agnostic version - doesn't require VAE model.

    Args:
        X_all: All images [N, C, H, W] (CPU tensor)
        Y_all: All labels [N] (numpy array)
        probs: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold for filtering
        out_dir: Output directory (Path or str)
        epoch: Current epoch (int or str)
        class_names: List of class names
        sample_topn: Number of samples to show per cluster
        nrow: Number of images per row in grid
    """
    import torchvision.utils as vutils

    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    # Handle string epochs by implementing the functionality directly
    if isinstance(epoch, str):
        # Implement the cluster sample grid saving directly
        N, K = probs.shape
        hard = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        for k in range(K):
            idx = np.where(hard == k)[0]

            if idx.size == 0:
                continue

            # Apply confidence threshold
            if conf_thresh > 0.0:
                idx = idx[conf[idx] >= conf_thresh]
                if idx.size == 0:
                    continue

            # Sort by confidence
            scores = probs[idx, k]
            order = np.argsort(-scores)
            idx = idx[order][:sample_topn]

            # Create grid
            grid = X_all[idx]
            save_path = out_dir / f"cluster_{k:02d}_samples_epoch_{epoch}.png"
            vutils.save_image(grid, str(save_path), nrow=nrow, normalize=True, value_range=(0, 1))
    else:
        # Use original function for integer epochs
        save_cluster_sample_grids(
            vae=None,  # Not used by the function
            X_all=X_all,
            Y_all=Y_all,
            prob=probs,
            conf_thresh=conf_thresh,
            out_dir=str(out_dir),
            epoch=epoch,
            class_names=class_names,
            sample_topn_per_cluster=sample_topn,
            nrow=nrow
        )


def build_and_save_atlas_A(X_all, probs, conf_thresh, gamma_power, atlas_robust,
                           sample_topn, max_per_cluster, out_dir, epoch):
    """
    Build and save pixel-space robust median/mean atlases.

    Model-agnostic - operates on pixel space directly.

    Args:
        X_all: All images [N, C, H, W] (CPU tensor)
        probs: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold
        gamma_power: Probability sharpening exponent
        atlas_robust: 'median' or 'mean'
        sample_topn: Max samples per cluster for atlas
        max_per_cluster: Hard limit on samples per cluster
        out_dir: Output directory (Path or str)
        epoch: Current epoch (int or str)
    """
    import torchvision.utils as vutils

    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    # Build atlases
    atlases = build_cluster_atlas_A(
        X_all=X_all,
        prob=probs,
        conf_thresh=conf_thresh,
        gamma_power=gamma_power,
        atlas_robust=atlas_robust,
        sample_topn=sample_topn,
        max_per_cluster=max_per_cluster
    )

    # Save atlases (handle string epochs)
    if isinstance(epoch, str):
        K = atlases.shape[0]
        # Save grid
        vutils.save_image(
            atlases,
            str(out_dir / f"atlas_epoch_{epoch}.png"),
            nrow=K,
            normalize=True,
            value_range=(0, 1)
        )
        # Save individual atlases
        for k in range(K):
            vutils.save_image(
                atlases[k],
                str(out_dir / f"atlas_cluster_{k:02d}_epoch_{epoch}.png"),
                normalize=True,
                value_range=(0, 1)
            )
    else:
        save_atlas_A_images(atlases, str(out_dir), epoch)


def build_and_save_atlas_laplacian(model, Z_all, probs, conf_thresh, gamma_power,
                                   knn_k, max_points, out_dir, epoch):
    """
    Build and save heat kernel manifold atlases.

    Requires model.decode() and model.latent_dim.

    Args:
        model: VAE model with decode() method
        Z_all: All latent codes [N, D] (numpy array)
        probs: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold
        gamma_power: Probability sharpening exponent
        knn_k: k for k-NN sigma estimation
        max_points: Max points per cluster
        out_dir: Output directory (Path or str)
        epoch: Current epoch (int or str)
    """
    import torchvision.utils as vutils

    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    # Build atlases
    atlases = build_cluster_atlas_laplacian(
        vae=model,
        Z_all=Z_all,
        prob=probs,
        conf_thresh=conf_thresh,
        gamma_power=gamma_power,
        knn_k=knn_k,
        geodesic_max_points=max_points
    )

    # Save atlases (handle string epochs)
    if isinstance(epoch, str):
        K = atlases.shape[0]
        # Save grid
        vutils.save_image(
            atlases,
            str(out_dir / f"atlas_epoch_{epoch}.png"),
            nrow=K,
            normalize=True,
            value_range=(0, 1)
        )
        # Save individual atlases
        for k in range(K):
            vutils.save_image(
                atlases[k],
                str(out_dir / f"atlas_cluster_{k:02d}_epoch_{epoch}.png"),
                normalize=True,
                value_range=(0, 1)
            )
    else:
        save_atlas_laplacian_images(atlases, str(out_dir), epoch)


def build_and_save_atlas_geodesic(model, Z_all, probs, conf_thresh, gamma_power,
                                  knn_k, max_points, out_dir, epoch, seed):
    """
    Build and save geodesic medoid manifold atlases.

    Requires model.decode() and model.latent_dim.

    Args:
        model: VAE model with decode() method
        Z_all: All latent codes [N, D] (numpy array)
        probs: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold
        gamma_power: Probability sharpening exponent
        knn_k: k for k-NN graph construction
        max_points: Max points per cluster
        out_dir: Output directory (Path or str)
        epoch: Current epoch (int or str)
        seed: Random seed for subsampling
    """
    import torchvision.utils as vutils

    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    # Build atlases
    atlases = build_cluster_atlas_geodesic_medoid(
        vae=model,
        Z_all=Z_all,
        prob=probs,
        conf_thresh=conf_thresh,
        gamma_power=gamma_power,
        knn_k=knn_k,
        geodesic_max_points=max_points,
        seed=seed,
        epoch=epoch if isinstance(epoch, int) else 0
    )

    # Save atlases (handle string epochs)
    if isinstance(epoch, str):
        K = atlases.shape[0]
        # Save grid
        vutils.save_image(
            atlases,
            str(out_dir / f"atlas_epoch_{epoch}.png"),
            nrow=K,
            normalize=True,
            value_range=(0, 1)
        )
        # Save individual atlases
        for k in range(K):
            vutils.save_image(
                atlases[k],
                str(out_dir / f"atlas_cluster_{k:02d}_epoch_{epoch}.png"),
                normalize=True,
                value_range=(0, 1)
            )
    else:
        save_atlas_geodesic_images(atlases, str(out_dir), epoch)


def save_gmm_means_atlas(model, gmm, out_dir, epoch, name='refit', nrow=3):
    """
    Decode and save GMM cluster means as atlases.

    Works for both refitted sklearn GMM and VAE prior means.

    Args:
        model: VAE model with decode() method
        gmm: Either sklearn GaussianMixture object or 'vae_prior' string
        out_dir: Output directory (Path or str)
        epoch: Current epoch (int or str)
        name: Name identifier ('refit' or 'vae_prior')
        nrow: Number of atlases per row in grid
    """
    import torchvision.utils as vutils

    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    device = next(model.parameters()).device
    model.eval()

    if gmm == 'vae_prior' or name == 'vae_prior':
        # Use VAE's trainable GMM prior means
        if isinstance(epoch, str):
            # Handle string epoch
            K = model.num_clusters
            means = model.gmm_means  # [K, D]

            atlases = []
            with torch.no_grad():
                for k in range(K):
                    z = means[k:k+1]  # [1, D]
                    atlas = model.decode(z).detach().cpu().squeeze(0)
                    atlases.append(atlas)

            atlases = torch.stack(atlases, dim=0)  # [K, C, H, W]

            # Save grid
            vutils.save_image(
                atlases,
                str(out_dir / f"vae_prior_means_decoded_epoch_{epoch}.png"),
                nrow=nrow,
                normalize=True,
                value_range=(0, 1)
            )

            # Save individual atlases
            for k in range(K):
                vutils.save_image(
                    atlases[k],
                    str(out_dir / f"vae_prior_mean_atlas_cluster_{k:02d}_epoch_{epoch}.png"),
                    normalize=True,
                    value_range=(0, 1)
                )
        else:
            save_vae_prior_means_atlas(model, str(out_dir), epoch, nrow=nrow)
    else:
        # Use refitted GMM means
        if isinstance(epoch, str):
            # Handle string epoch
            K = gmm.n_components
            means = torch.from_numpy(gmm.means_).float().to(device)  # [K, D]

            atlases = []
            with torch.no_grad():
                for k in range(K):
                    z = means[k:k+1]  # [1, D]
                    atlas = model.decode(z).detach().cpu().squeeze(0)
                    atlases.append(atlas)

            atlases = torch.stack(atlases, dim=0)  # [K, C, H, W]

            # Save grid
            vutils.save_image(
                atlases,
                str(out_dir / f"gmm_refit_means_atlas_epoch_{epoch}.png"),
                nrow=nrow,
                normalize=True,
                value_range=(0, 1)
            )

            # Save individual atlases
            for k in range(K):
                vutils.save_image(
                    atlases[k],
                    str(out_dir / f"gmm_refit_mean_atlas_cluster_{k:02d}_epoch_{epoch}.png"),
                    normalize=True,
                    value_range=(0, 1)
                )
        else:
            save_refit_gmm_means_atlas(model, gmm, str(out_dir), epoch, nrow=nrow)


def compute_and_save_cluster_purity(cluster_assignments, true_labels, num_clusters,
                                    class_names, out_dir, epoch):
    """
    Compute cluster purity statistics and save to CSV.

    Args:
        cluster_assignments: Predicted cluster labels [N] (numpy array)
        true_labels: Ground truth class labels [N] (numpy array)
        num_clusters: Number of clusters (int)
        class_names: List of class names
        out_dir: Output directory (Path or str)
        epoch: Current epoch (int or str)
    """
    out_dir = Path(out_dir)

    # Compute purity statistics
    purity_df = compute_cluster_purity(
        cluster_assignments=cluster_assignments,
        true_labels=true_labels,
        num_clusters=num_clusters,
        class_names=class_names
    )

    # Save to CSV
    if isinstance(epoch, int):
        csv_path = out_dir / f'cluster_purity_epoch_{epoch:03d}.csv'
    else:
        csv_path = out_dir / f'cluster_purity_{epoch}.csv'

    purity_df.to_csv(csv_path, index=False)
