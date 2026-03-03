"""
Atlas generation and cluster analysis utilities for CLAST.

This module contains functions for:
- Pixel-space robust median atlases (Scheme-A)
- Heat kernel (Laplacian) atlases
- Geodesic medoid atlases
- Cluster purity analysis
- Cluster sample visualization
"""

import os
import numpy as np
import torch
import torchvision.utils as vutils
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


# =========================
# Helper Functions
# =========================

def sharpen_probs(P, gamma_power=2.0):
    """
    Sharpen probability distribution using power transform.

    Args:
        P: Probability matrix [N, K]
        gamma_power: Sharpening exponent (>1 sharpens, <1 smooths)

    Returns:
        Sharpened probabilities [N, K]
    """
    if gamma_power == 1.0:
        return P
    P_sharp = np.power(P, gamma_power)
    P_sharp = P_sharp / (P_sharp.sum(axis=1, keepdims=True) + 1e-12)
    return P_sharp


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# =========================
# Cluster Purity Analysis
# =========================

def compute_cluster_purity(cluster_assignments, true_labels, num_clusters, class_names=None):
    """
    Compute cluster purity statistics.

    Args:
        cluster_assignments: Predicted cluster labels [N]
        true_labels: Ground truth class labels [N]
        num_clusters: Number of clusters (K)
        class_names: List of class names (optional)

    Returns:
        DataFrame with columns: cluster, count, top_class, purity, class_*_count
    """
    # Handle unsupervised case (all labels = -1)
    unique_labels = set(true_labels)
    if unique_labels == {-1} or all(label < 0 for label in unique_labels):
        # No ground truth labels - return simple cluster counts without purity
        stats = []
        for k in range(num_clusters):
            idx = np.where(cluster_assignments == k)[0]
            stats.append({
                "cluster": k,
                "count": len(idx),
                "top_class": -1,
                "purity": float('nan')  # N/A for unsupervised
            })
        return pd.DataFrame(stats)

    num_classes = len(set(true_labels))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    stats = []
    for k in range(num_clusters):
        idx = np.where(cluster_assignments == k)[0]

        if idx.size == 0:
            row = {"cluster": k, "count": 0, "top_class": -1, "purity": 0.0}
            for c, name in enumerate(class_names):
                row[f"class_{name}_count"] = 0
            stats.append(row)
            continue

        labels_k = true_labels[idx]
        counts = np.bincount(labels_k, minlength=num_classes)
        top_class = int(counts.argmax())
        purity = float(counts.max() / counts.sum())

        row = {
            "cluster": k,
            "count": int(counts.sum()),
            "top_class": top_class,
            "purity": purity
        }
        for c, name in enumerate(class_names):
            row[f"class_{name}_count"] = int(counts[c])

        stats.append(row)

    return pd.DataFrame(stats)


# =========================
# Cluster Sample Visualization
# =========================

@torch.no_grad()
def save_cluster_sample_grids(vae, X_all, Y_all, prob, conf_thresh, out_dir, epoch,
                               class_names, sample_topn_per_cluster=32, nrow=8):
    """
    Save grid of top samples per cluster.

    Args:
        vae: VAE model (not used, kept for interface compatibility)
        X_all: All images [N, C, H, W] (CPU tensor)
        Y_all: All labels [N]
        prob: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold for filtering
        out_dir: Output directory
        epoch: Current epoch
        class_names: List of class names
        sample_topn_per_cluster: Number of samples to show per cluster
        nrow: Number of images per row in grid
    """
    ensure_dir(out_dir)

    N, K = prob.shape
    hard = prob.argmax(axis=1)
    conf = prob.max(axis=1)

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
        scores = prob[idx, k]
        order = np.argsort(-scores)
        idx = idx[order][:sample_topn_per_cluster]

        # Create grid
        grid = X_all[idx]
        save_path = os.path.join(out_dir, f"cluster_{k:02d}_samples_epoch_{epoch:03d}.png")
        vutils.save_image(grid, save_path, nrow=nrow, normalize=True, value_range=(0, 1))


# =========================
# Scheme-A: Pixel-Space Robust Median Atlas
# =========================

@torch.no_grad()
def build_cluster_atlas_A(X_all, prob, conf_thresh, gamma_power, atlas_robust,
                          sample_topn, max_per_cluster):
    """
    Build pixel-space robust median (or mean) atlases.

    Args:
        X_all: All images [N, C, H, W] (torch tensor on CPU)
        prob: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold
        gamma_power: Probability sharpening exponent
        atlas_robust: 'median' or 'mean'
        sample_topn: Max samples per cluster for atlas
        max_per_cluster: Hard limit on samples per cluster

    Returns:
        atlases: Torch tensor [K, C, H, W]
    """
    P = sharpen_probs(prob, gamma_power)
    hard = P.argmax(axis=1)
    conf = P.max(axis=1)
    N, K = P.shape

    # Detect image size from first image
    C, H, W = int(X_all.shape[1]), int(X_all.shape[2]), int(X_all.shape[3])

    atlases = []
    for k in range(K):
        idx = np.where(hard == k)[0]

        if idx.size == 0:
            atlas = torch.zeros(C, H, W)
        else:
            # Apply confidence threshold
            if conf_thresh > 0.0:
                idx = idx[conf[idx] >= conf_thresh]

            if idx.size == 0:
                atlas = torch.zeros(C, H, W)
            else:
                # Limit number of samples
                if idx.size > max_per_cluster:
                    order = np.argsort(-P[idx, k])
                    idx = idx[order][:max_per_cluster]

                imgs = X_all[idx].float()

                if atlas_robust == "median":
                    atlas = imgs.median(dim=0).values
                else:
                    atlas = imgs.mean(dim=0)

        atlases.append(atlas)

    atlases = torch.stack(atlases, dim=0)
    return atlases


def save_atlas_A_images(atlases, out_dir, epoch):
    """Save Scheme-A atlas images."""
    ensure_dir(out_dir)
    K = atlases.shape[0]

    # Save grid
    vutils.save_image(
        atlases,
        os.path.join(out_dir, f"atlas_epoch_{epoch:03d}.png"),
        nrow=K,
        normalize=True,
        value_range=(0, 1)
    )

    # Save individual atlases
    for k in range(K):
        vutils.save_image(
            atlases[k],
            os.path.join(out_dir, f"atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
            normalize=True,
            value_range=(0, 1)
        )


# =========================
# Heat Kernel (Laplacian) Atlas
# =========================

@torch.no_grad()
def build_cluster_atlas_laplacian(vae, Z_all, prob, conf_thresh, gamma_power,
                                   knn_k, geodesic_max_points):
    """
    Build latent-manifold atlases using heat kernel (RBF kernel density).

    For each cluster:
    1. Compute RBF kernel weights between all points
    2. Select point with highest local density (sum of weights)
    3. Decode to get atlas

    Args:
        vae: VAE model
        Z_all: All latent codes [N, D] (numpy array)
        prob: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold
        gamma_power: Probability sharpening exponent
        knn_k: k for k-NN sigma estimation
        geodesic_max_points: Max points per cluster

    Returns:
        atlases: Torch tensor [K, C, H, W]
    """
    P = sharpen_probs(prob, gamma_power)
    hard = P.argmax(axis=1)
    conf = P.max(axis=1)
    N, K = P.shape

    device = next(vae.parameters()).device
    vae.eval()

    # Detect image size from decoding a dummy latent
    with torch.no_grad():
        dummy_z = torch.zeros(1, vae.latent_dim, device=device)
        dummy_out = vae.decode(dummy_z)
        # dummy_out shape: [batch=1, C, H, W]
        _, C, H, W = dummy_out.shape
        C, H, W = int(C), int(H), int(W)

    atlases = []
    for k in range(K):
        idx = np.where(hard == k)[0]

        if idx.size == 0:
            atlas = torch.zeros(C, H, W)
            atlases.append(atlas)
            continue

        # Apply confidence threshold
        if conf_thresh > 0.0:
            idx = idx[conf[idx] >= conf_thresh]

        if idx.size == 0:
            atlas = torch.zeros(C, H, W)
            atlases.append(atlas)
            continue

        # Subsample if too many points
        if idx.size > geodesic_max_points:
            scores = P[idx, k]
            order = np.argsort(-scores)
            idx = idx[order[:geodesic_max_points]]

        Zk = Z_all[idx]  # [M, D]
        M = Zk.shape[0]

        if M <= 1:
            # Single point - just decode it
            z_med = torch.tensor(Zk[0], dtype=torch.float32, device=device).unsqueeze(0)
            atlas = vae.decode(z_med).detach().cpu().squeeze(0)
            atlases.append(atlas)
            continue

        # Estimate sigma using k-NN
        k_eff = min(knn_k, M - 1)
        nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(Zk)
        dists, _ = nbrs.kneighbors(Zk)
        sigma = np.median(dists[:, -1])  # Use k-th nearest neighbor distance

        if sigma < 1e-8:
            sigma = 1.0  # Fallback

        # Compute RBF kernel weights
        W_kernel = rbf_kernel(Zk, gamma=1.0 / (2 * sigma ** 2))

        # Find point with highest local density
        densities = W_kernel.sum(axis=1)
        med_local = int(np.argmax(densities))

        # Decode medoid
        z_med = torch.tensor(Zk[med_local], dtype=torch.float32, device=device).unsqueeze(0)
        atlas = vae.decode(z_med).detach().cpu().squeeze(0)

        atlases.append(atlas)

    atlases = torch.stack(atlases, dim=0)
    return atlases


def save_atlas_laplacian_images(atlases, out_dir, epoch):
    """Save Laplacian atlas images."""
    ensure_dir(out_dir)
    K = atlases.shape[0]

    # Save grid
    vutils.save_image(
        atlases,
        os.path.join(out_dir, f"atlas_epoch_{epoch:03d}.png"),
        nrow=K,
        normalize=True,
        value_range=(0, 1)
    )

    # Save individual atlases
    for k in range(K):
        vutils.save_image(
            atlases[k],
            os.path.join(out_dir, f"atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
            normalize=True,
            value_range=(0, 1)
        )


# =========================
# Geodesic Medoid Atlas
# =========================

@torch.no_grad()
def build_cluster_atlas_geodesic_medoid(vae, Z_all, prob, conf_thresh, gamma_power,
                                        knn_k, geodesic_max_points, seed=42, epoch=0):
    """
    Build latent-manifold atlases using geodesic medoid (shortest path on k-NN graph).

    For each cluster:
    1. Build k-NN graph with Euclidean edge weights
    2. Compute all-pairs shortest path (Dijkstra)
    3. Select medoid: argmin sum of geodesic distances
    4. Decode to get atlas

    Args:
        vae: VAE model
        Z_all: All latent codes [N, D] (numpy array)
        prob: Cluster probabilities [N, K] (numpy array)
        conf_thresh: Confidence threshold
        gamma_power: Probability sharpening exponent
        knn_k: k for k-NN graph construction
        geodesic_max_points: Max points per cluster
        seed: Random seed for subsampling
        epoch: Current epoch (for seed variation)

    Returns:
        atlases: Torch tensor [K, C, H, W]
    """
    P = sharpen_probs(prob, gamma_power)
    hard = P.argmax(axis=1)
    conf = P.max(axis=1)
    N, K = P.shape

    device = next(vae.parameters()).device
    vae.eval()

    rng = np.random.default_rng(seed + epoch)

    # Detect image size
    with torch.no_grad():
        dummy_z = torch.zeros(1, vae.latent_dim, device=device)
        dummy_out = vae.decode(dummy_z)
        # dummy_out shape: [batch=1, C, H, W]
        _, C, H, W = dummy_out.shape
        C, H, W = int(C), int(H), int(W)

    atlases = []
    for k in range(K):
        idx = np.where(hard == k)[0]

        if idx.size == 0:
            atlas = torch.zeros(C, H, W)
            atlases.append(atlas)
            continue

        # Apply confidence threshold
        if conf_thresh > 0.0:
            idx = idx[conf[idx] >= conf_thresh]

        if idx.size == 0:
            atlas = torch.zeros(C, H, W)
            atlases.append(atlas)
            continue

        # Subsample if too many points (keep high-confidence + random)
        if idx.size > geodesic_max_points:
            scores = P[idx, k]
            order = np.argsort(-scores)
            top_half = geodesic_max_points // 2
            top = idx[order[:top_half]]
            rest = idx[order[top_half:]]

            if rest.size > 0:
                m = geodesic_max_points - top.size
                take = rng.choice(rest, size=m, replace=False)
                idx = np.concatenate([top, take])
            else:
                idx = top

        Zk = Z_all[idx]  # [M, D]
        M = Zk.shape[0]

        if M <= 2:
            # Too few points - just use highest confidence
            best_local = int(np.argmax(P[idx, k]))
            z_med = torch.tensor(Zk[best_local], dtype=torch.float32, device=device).unsqueeze(0)
            atlas = vae.decode(z_med).detach().cpu().squeeze(0)
            atlases.append(atlas)
            continue

        # Build k-NN graph
        k_eff = int(min(max(2, knn_k), M - 1))
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(Zk)
        dists, neigh = nbrs.kneighbors(Zk)  # [M, k+1]

        # Remove self (first column)
        dists = dists[:, 1:]
        neigh = neigh[:, 1:]

        # Build sparse adjacency matrix
        rows = np.repeat(np.arange(M), k_eff)
        cols = neigh.reshape(-1)
        data = dists.reshape(-1)

        # Make symmetric (undirected graph)
        rows2 = np.concatenate([rows, cols])
        cols2 = np.concatenate([cols, rows])
        data2 = np.concatenate([data, data])

        G = csr_matrix((data2, (rows2, cols2)), shape=(M, M))

        # All-pairs shortest path
        Dg = shortest_path(G, directed=False, unweighted=False)

        # Handle disconnected components
        inf_mask = ~np.isfinite(Dg)
        if np.any(inf_mask):
            finite_max = np.max(Dg[np.isfinite(Dg)]) if np.any(np.isfinite(Dg)) else 1.0
            Dg[inf_mask] = 10.0 * finite_max

        # Find geodesic medoid
        s = Dg.sum(axis=1)
        med_local = int(np.argmin(s))

        # Decode medoid
        z_med = torch.tensor(Zk[med_local], dtype=torch.float32, device=device).unsqueeze(0)
        atlas = vae.decode(z_med).detach().cpu().squeeze(0)

        atlases.append(atlas)

    atlases = torch.stack(atlases, dim=0)
    return atlases


def save_atlas_geodesic_images(atlases, out_dir, epoch):
    """Save geodesic medoid atlas images."""
    ensure_dir(out_dir)
    K = atlases.shape[0]

    # Save grid
    vutils.save_image(
        atlases,
        os.path.join(out_dir, f"atlas_epoch_{epoch:03d}.png"),
        nrow=K,
        normalize=True,
        value_range=(0, 1)
    )

    # Save individual atlases
    for k in range(K):
        vutils.save_image(
            atlases[k],
            os.path.join(out_dir, f"atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
            normalize=True,
            value_range=(0, 1)
        )


# =========================
# Refit GMM Means Atlas
# =========================

@torch.no_grad()
def save_refit_gmm_means_atlas(vae, gmm, out_dir, epoch, nrow=3):
    """
    Save atlases decoded from refitted GMM means.

    Args:
        vae: VAE model
        gmm: Fitted sklearn GaussianMixture
        out_dir: Output directory
        epoch: Current epoch
        nrow: Number of atlases per row in grid
    """
    ensure_dir(out_dir)

    device = next(vae.parameters()).device
    vae.eval()

    K = gmm.n_components
    means = torch.from_numpy(gmm.means_).float().to(device)  # [K, D]

    atlases = []
    for k in range(K):
        z = means[k:k+1]  # [1, D]
        atlas = vae.decode(z).detach().cpu().squeeze(0)
        atlases.append(atlas)

    atlases = torch.stack(atlases, dim=0)  # [K, C, H, W]

    # Save grid
    vutils.save_image(
        atlases,
        os.path.join(out_dir, f"gmm_refit_means_atlas_epoch_{epoch:03d}.png"),
        nrow=nrow,
        normalize=True,
        value_range=(0, 1)
    )

    # Save individual atlases
    for k in range(K):
        vutils.save_image(
            atlases[k],
            os.path.join(out_dir, f"gmm_refit_mean_atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
            normalize=True,
            value_range=(0, 1)
        )


# =========================
# VAE Prior Means Atlas
# =========================

@torch.no_grad()
def save_vae_prior_means_atlas(vae, out_dir, epoch, nrow=3):
    """
    Save atlases decoded from VAE's trainable GMM prior means.

    Args:
        vae: VAE model
        out_dir: Output directory
        epoch: Current epoch
        nrow: Number of atlases per row in grid
    """
    ensure_dir(out_dir)

    device = next(vae.parameters()).device
    vae.eval()

    K = vae.num_clusters
    means = vae.gmm_means  # [K, D]

    atlases = []
    for k in range(K):
        z = means[k:k+1]  # [1, D]
        atlas = vae.decode(z).detach().cpu().squeeze(0)
        atlases.append(atlas)

    atlases = torch.stack(atlases, dim=0)  # [K, C, H, W]

    # Save grid
    vutils.save_image(
        atlases,
        os.path.join(out_dir, f"vae_prior_means_decoded_epoch_{epoch:03d}.png"),
        nrow=nrow,
        normalize=True,
        value_range=(0, 1)
    )

    # Save individual atlases (optional)
    for k in range(K):
        vutils.save_image(
            atlases[k],
            os.path.join(out_dir, f"vae_prior_mean_atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
            normalize=True,
            value_range=(0, 1)
        )
