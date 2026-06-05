"""
Unified evaluation script for all model types.

Loads a trained model checkpoint, evaluates clustering metrics on the test set,
and generates visualizations (cluster means, reconstructions, t-SNE, purity).

Supports:
  - vae_gmm: VAE with Gaussian Mixture Model prior
  - ours: Our manifold-aware atlas clustering model
  - diffusion_vae: VAE-GMM with latent-space diffusion denoiser
  - baseline_gmm: Pixel-space GMM clustering (non-neural)
  - baseline_kmeans: Pixel-space K-Means clustering (non-neural)

Usage:
  python test.py --config configs/mnist/vae_gmm.yaml \
      --checkpoint checkpoints/mnist/vae_gmm/checkpoint_best.pth \
      --output_dir results/test_vae_gmm/

  # With MC uncertainty estimation
  python test.py --config configs/cardiac/ours.yaml \
      --checkpoint checkpoints/cardiac/ours/checkpoint_best.pth \
      --output_dir results/test_ours/ \
      --uncertainty --n_mc 30

  # Override GMM covariance type
  python test.py --config configs/mnist/vae_gmm.yaml \
      --checkpoint checkpoints/mnist/vae_gmm/checkpoint_best.pth \
      --output_dir results/test_vae_gmm/ \
      --covariance_type diag
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from collections import Counter

from utils.config import load_yaml_config, get_default_config
from utils.data_loader import get_data_loaders
from utils.metrics import (
    cluster_accuracy,
    compute_metrics,
    compute_unsupervised_metrics,
    reduce_dimensionality,
)
from utils.visualization import plot_reconstructions, plot_latent_space


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified evaluation for VAE-GMM, ours, Diffusion-VAE, and baseline models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation outputs')

    # GMM settings
    parser.add_argument('--covariance_type', type=str, default='full',
                        choices=['full', 'diag', 'spherical'],
                        help='GMM covariance type for evaluation')

    # MC uncertainty
    parser.add_argument('--uncertainty', action='store_true',
                        help='Enable MC dropout / reparameterization uncertainty estimation')
    parser.add_argument('--n_mc', type=int, default=30,
                        help='Number of MC forward passes for uncertainty estimation')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g., cuda:0, cpu)')

    return parser.parse_args()


# =============================================================================
# Model loading
# =============================================================================

def detect_model_type(config):
    """Detect model type from config['model']['name']."""
    model_name = config['model']['name']
    if 'diffusion_vae' in model_name:
        return 'diffusion_vae'
    elif 'ours' in model_name:
        return 'ours'
    elif 'vae_gmm' in model_name:
        return 'vae_gmm'
    elif 'baseline_gmm' in model_name:
        return 'baseline_gmm'
    elif 'baseline_kmeans' in model_name:
        return 'baseline_kmeans'
    else:
        raise ValueError(f"Unknown model type in config: {model_name}")


def load_model(config, model_type, checkpoint_path, device):
    """Instantiate model from config and load checkpoint weights."""
    if model_type == 'vae_gmm':
        from models.vae_gmm import get_model
    elif model_type == 'ours':
        from models.ours import get_model
    elif model_type == 'diffusion_vae':
        from models.diffusion_vae import get_model
    elif model_type == 'baseline_gmm':
        from models.baseline_gmm import get_model
    elif model_type == 'baseline_kmeans':
        from models.baseline_kmeans import get_model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = get_model(config)

    # Baselines have no learned parameters to load from a checkpoint
    is_baseline = model_type in ('baseline_gmm', 'baseline_kmeans')

    if not is_baseline:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def is_neural_model(model_type):
    """Return True for models that have encoder/decoder (not pixel-space baselines)."""
    return model_type in ('vae_gmm', 'ours', 'diffusion_vae')


def is_supervised_dataset(dataset_name):
    """Return True if the dataset has ground-truth class labels."""
    return dataset_name in ('mnist', 'cardiac')


# =============================================================================
# Feature extraction
# =============================================================================

@torch.no_grad()
def extract_features(model, model_type, data_loader, device):
    """
    Extract features and labels from the dataset.

    For neural models: extract encoder mu (latent codes).
    For baselines: extract flattened pixel features.

    Returns:
        z_data: Feature array (N, D), lexicographically sorted
        labels: Label array (N,), sorted in same order
        images: Raw images (N, C, H, W) tensor, sorted in same order (neural only, else None)
    """
    if is_neural_model(model_type):
        model.eval()
        z_list, labels_list, images_list = [], [], []

        for batch, target in data_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            z_list.append(mu.cpu().numpy())
            labels_list.append(target.numpy())
            images_list.append(batch.cpu())

        z_data = np.vstack(z_list)
        labels = np.hstack(labels_list)
        images = torch.cat(images_list, dim=0)

        # Lexicographic sort for deterministic GMM fitting
        sort_idx = np.lexsort(z_data.T[::-1])
        z_data = z_data[sort_idx]
        labels = labels[sort_idx]
        images = images[sort_idx]

        return z_data, labels, images
    else:
        # Baselines use their own extract_latent_features method
        z_data, labels = model.extract_latent_features(data_loader, device)
        return z_data, labels, None


# =============================================================================
# GMM fitting and cluster assignment
# =============================================================================

def fit_gmm(z_data, num_clusters, covariance_type='full', n_init=10):
    """Fit sklearn GMM on feature array and return GMM + cluster labels."""
    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=0,
        reg_covar=1e-4,
    )
    cluster_labels = gmm.fit_predict(z_data)
    return gmm, cluster_labels


# =============================================================================
# Visualization helpers
# =============================================================================

def save_cluster_means(model, model_type, gmm, num_clusters, dataset_name, output_dir, device):
    """
    Decode GMM means and save as an image grid.

    For neural models: pass GMM means through decoder.
    For baselines: reshape centroids directly.
    """
    if is_neural_model(model_type):
        means_tensor = torch.from_numpy(gmm.means_.astype(np.float32)).to(device)
        decoded = []
        with torch.no_grad():
            for k in range(num_clusters):
                img = model.decode(means_tensor[k:k + 1]).cpu().squeeze().numpy()
                decoded.append(img)
        means_images = np.stack(decoded, axis=0)  # (K, H, W)
    else:
        # Baselines: GMM means are in normalized pixel space - reshape
        # model.scaler was used during feature extraction; inverse-transform
        means_pixel = model.scaler.inverse_transform(gmm.means_)
        image_size = model.image_size
        means_images = means_pixel.reshape(-1, image_size, image_size)

    # Plot grid
    K = means_images.shape[0]
    n_cols = min(K, 10)
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.2 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for ki in range(K):
        r, c = divmod(ki, n_cols)
        ax = axes[r, c]
        ax.imshow(np.clip(means_images[ki], 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'C{ki}', fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for ki in range(K, n_rows * n_cols):
        r, c = divmod(ki, n_cols)
        axes[r, c].axis('off')

    plt.suptitle('GMM Cluster Means', fontsize=12)
    plt.tight_layout()
    save_path = Path(output_dir) / 'cluster_means.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_reconstructions(model, images, output_dir, device, n_samples=10):
    """Save input vs reconstruction comparison for a batch (neural models only)."""
    n_samples = min(n_samples, images.shape[0])
    batch = images[:n_samples].to(device)

    with torch.no_grad():
        recon, _, _, _ = model(batch)

    save_path = Path(output_dir) / 'reconstructions.png'
    plot_reconstructions(batch, recon, str(save_path), n_samples=n_samples)
    return save_path


def save_latent_tsne(z_data, labels, class_names, output_dir, color_by='label'):
    """Save t-SNE visualization of latent space."""
    save_path = Path(output_dir) / 'latent_tsne.png'

    # Subsample for speed if very large
    max_points = 10000
    if z_data.shape[0] > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(z_data.shape[0], max_points, replace=False)
        z_sub = z_data[idx]
        labels_sub = labels[idx]
    else:
        z_sub = z_data
        labels_sub = labels

    plot_latent_space(z_sub, labels_sub, class_names, str(save_path), method='tsne')
    return save_path


def save_cluster_purity(labels, cluster_labels, num_clusters, class_names, output_dir):
    """
    Save per-cluster label distribution as CSV (supervised datasets only).

    Columns: cluster_id, dominant_label, dominant_count, total, purity, <per-class counts>
    """
    save_path = Path(output_dir) / 'cluster_purity.csv'

    rows = []
    for k in range(num_clusters):
        mask = cluster_labels == k
        total = int(mask.sum())
        if total == 0:
            row = {'cluster_id': k, 'dominant_label': -1, 'dominant_count': 0,
                   'total': 0, 'purity': 0.0}
            for cn in class_names:
                row[cn] = 0
            rows.append(row)
            continue

        cluster_true = labels[mask]
        counter = Counter(cluster_true.tolist())
        dominant_label = counter.most_common(1)[0][0]
        dominant_count = counter.most_common(1)[0][1]

        row = {
            'cluster_id': k,
            'dominant_label': dominant_label,
            'dominant_count': dominant_count,
            'total': total,
            'purity': dominant_count / total,
        }
        for ci, cn in enumerate(class_names):
            row[cn] = counter.get(ci, 0)
        rows.append(row)

    # Write CSV
    if rows:
        header = list(rows[0].keys())
        with open(save_path, 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(str(row[h]) for h in header) + '\n')

    return save_path


# =============================================================================
# MC Uncertainty estimation
# =============================================================================

@torch.no_grad()
def estimate_uncertainty(model, model_type, z_data, gmm, device, n_mc=30, num_clusters=10):
    """
    Run N_mc reparameterized forward passes through the decoder and compute
    per-pixel standard deviation for each cluster mean.

    Returns:
        uncertainty_maps: (K, H, W) per-pixel std across MC samples
    """
    means_tensor = torch.from_numpy(gmm.means_.astype(np.float32)).to(device)

    # Use the GMM covariances to sample around each mean
    # For simplicity, use diagonal approximation
    if gmm.covariance_type == 'full':
        covariances = np.stack([np.diag(c) for c in gmm.covariances_], axis=0)
    elif gmm.covariance_type == 'diag':
        covariances = gmm.covariances_
    elif gmm.covariance_type == 'spherical':
        D = gmm.means_.shape[1]
        covariances = np.repeat(gmm.covariances_[:, None], D, axis=1)
    else:
        covariances = np.ones_like(gmm.means_) * 0.01

    stds_tensor = torch.from_numpy(np.sqrt(covariances + 1e-8).astype(np.float32)).to(device)

    uncertainty_maps = []
    for k in range(num_clusters):
        samples = []
        mean_k = means_tensor[k:k + 1]  # (1, D)
        std_k = stds_tensor[k:k + 1]    # (1, D)

        for _ in range(n_mc):
            eps = torch.randn_like(mean_k)
            z_sample = mean_k + eps * std_k
            decoded = model.decode(z_sample).cpu().squeeze().numpy()  # (H, W)
            samples.append(decoded)

        samples = np.stack(samples, axis=0)  # (n_mc, H, W)
        std_map = np.std(samples, axis=0)    # (H, W)
        uncertainty_maps.append(std_map)

    return np.stack(uncertainty_maps, axis=0)  # (K, H, W)


def save_uncertainty_heatmap(uncertainty_maps, output_dir, num_clusters):
    """Save uncertainty heatmap visualization."""
    K = uncertainty_maps.shape[0]
    n_cols = min(K, 10)
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    vmax = np.percentile(uncertainty_maps, 99)

    for ki in range(K):
        r, c = divmod(ki, n_cols)
        ax = axes[r, c]
        im = ax.imshow(uncertainty_maps[ki], cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'C{ki}', fontsize=8)
        ax.axis('off')

    for ki in range(K, n_rows * n_cols):
        r, c = divmod(ki, n_cols)
        axes[r, c].axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Pixel Std')
    plt.suptitle('MC Uncertainty (per-pixel std)', fontsize=12)
    plt.tight_layout()
    save_path = Path(output_dir) / 'uncertainty_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# =============================================================================
# Diffusion-VAE specific evaluation
# =============================================================================

def evaluate_diffusion_denoised(model, config, data_loader, device, num_clusters,
                                covariance_type, output_dir, labels, supervised):
    """
    For diffusion_vae models: evaluate denoised latents at multiple timesteps.

    Returns dict of {timestep: metrics_dict}.
    """
    diffusion_config = config.get('diffusion', {})
    eval_timesteps = diffusion_config.get('eval_timesteps', [1, 2, 3, 5])
    n_mc = diffusion_config.get('n_mc', 50)

    results = {}
    for t in eval_timesteps:
        z_denoised, labels_sorted = model.extract_denoised_latent_features(
            data_loader, device, denoise_t=t, n_mc=n_mc
        )

        gmm, cluster_labels = fit_gmm(z_denoised, num_clusters, covariance_type)

        if supervised:
            metrics = compute_metrics(labels_sorted, cluster_labels)
        else:
            metrics = compute_unsupervised_metrics(z_denoised, cluster_labels)

        results[t] = metrics
        print(f"  Denoised t={t}: {metrics}")

    # Save denoised metrics
    denoised_path = Path(output_dir) / 'denoised_metrics.yaml'
    denoised_out = {}
    for t, m in results.items():
        denoised_out[f't_{t}'] = {k: float(v) for k, v in m.items()}
    with open(denoised_path, 'w') as f:
        yaml.dump(denoised_out, f, default_flow_style=False, sort_keys=False)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t_start = time.time()

    # ── Output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──
    config = load_yaml_config(args.config)
    # Fill missing keys from defaults
    default_config = get_default_config()
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
        elif isinstance(default_config[key], dict):
            for subkey in default_config[key]:
                if subkey not in config[key]:
                    config[key][subkey] = default_config[key][subkey]

    # ── Device ──
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device_str = config['training'].get('device', 'cuda')
        if device_str == 'cuda' and not torch.cuda.is_available():
            device_str = 'cpu'
        device = torch.device(device_str)

    # ── Detect model type ──
    model_type = detect_model_type(config)
    dataset_name = config['dataset']['name']
    num_clusters = config['model']['num_clusters']
    supervised = is_supervised_dataset(dataset_name)
    neural = is_neural_model(model_type)

    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print(f"  Model type:       {model_type}")
    print(f"  Dataset:          {dataset_name}")
    print(f"  Num clusters:     {num_clusters}")
    print(f"  Supervised:       {supervised}")
    print(f"  Neural model:     {neural}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Covariance type:  {args.covariance_type}")
    print(f"  Output dir:       {output_dir}")
    if args.uncertainty:
        print(f"  Uncertainty:      ON (n_mc={args.n_mc})")
    print("=" * 70)

    # ── Load model ──
    print("\nLoading model ...")
    model = load_model(config, model_type, args.checkpoint, device)

    # ── Load data ──
    print("\nLoading data ...")
    # Use shuffle=False for deterministic evaluation
    eval_config = dict(config)
    eval_config['dataset'] = dict(config['dataset'])
    eval_config['dataset']['shuffle'] = False
    eval_config['dataset']['num_workers'] = 0

    train_loader, test_loader, dataset_info = get_data_loaders(eval_config)
    class_names = dataset_info['class_names']

    # ── Extract features (test set) ──
    print("\nExtracting features from test set ...")
    z_data, labels, images = extract_features(model, model_type, test_loader, device)
    print(f"  Features shape: {z_data.shape}")
    print(f"  Labels shape:   {labels.shape}")

    # ── Fit GMM ──
    print(f"\nFitting GMM (K={num_clusters}, covariance={args.covariance_type}) ...")
    gmm, cluster_labels = fit_gmm(z_data, num_clusters, args.covariance_type)

    # ── Compute metrics ──
    print("\nComputing metrics ...")
    all_metrics = {}

    if supervised:
        sup_metrics = compute_metrics(labels, cluster_labels)
        all_metrics.update(sup_metrics)
        print(f"  Accuracy (Hungarian): {sup_metrics['accuracy']:.4f}")
        print(f"  NMI:                  {sup_metrics['nmi']:.4f}")
        print(f"  ARI:                  {sup_metrics['ari']:.4f}")
    else:
        unsup_metrics = compute_unsupervised_metrics(z_data, cluster_labels)
        all_metrics.update(unsup_metrics)
        print(f"  Silhouette:        {unsup_metrics['silhouette']:.4f}")
        print(f"  Calinski-Harabasz: {unsup_metrics['calinski_harabasz']:.4f}")
        print(f"  Davies-Bouldin:    {unsup_metrics['davies_bouldin']:.4f}")

    all_metrics['num_samples'] = int(z_data.shape[0])
    all_metrics['num_clusters'] = num_clusters
    all_metrics['covariance_type'] = args.covariance_type
    all_metrics['model_type'] = model_type
    all_metrics['dataset'] = dataset_name

    # ── Visualizations ──
    print("\nGenerating visualizations ...")

    # 1. Cluster means
    print("  - cluster_means.png")
    save_cluster_means(model, model_type, gmm, num_clusters, dataset_name, output_dir, device)

    # 2. Reconstructions (neural only)
    if neural and images is not None:
        print("  - reconstructions.png")
        save_reconstructions(model, images, output_dir, device)

    # 3. Latent t-SNE
    print("  - latent_tsne.png")
    if supervised:
        save_latent_tsne(z_data, labels, class_names, output_dir, color_by='label')
    else:
        # Color by cluster assignment for unsupervised
        cluster_names = [f'Cluster {i}' for i in range(num_clusters)]
        save_latent_tsne(z_data, cluster_labels, cluster_names, output_dir, color_by='cluster')

    # 4. Cluster purity (supervised only)
    if supervised:
        print("  - cluster_purity.csv")
        save_cluster_purity(labels, cluster_labels, num_clusters, class_names, output_dir)

    # ── MC Uncertainty ──
    if args.uncertainty and neural:
        print(f"\nEstimating MC uncertainty (n_mc={args.n_mc}) ...")
        uncertainty_maps = estimate_uncertainty(
            model, model_type, z_data, gmm, device,
            n_mc=args.n_mc, num_clusters=num_clusters
        )
        save_uncertainty_heatmap(uncertainty_maps, output_dir, num_clusters)
        print("  - uncertainty_heatmap.png saved")

        # Record mean uncertainty per cluster
        for k in range(num_clusters):
            all_metrics[f'uncertainty_mean_c{k}'] = float(np.mean(uncertainty_maps[k]))
        all_metrics['uncertainty_mean_overall'] = float(np.mean(uncertainty_maps))

    # ── Diffusion-VAE specific: evaluate denoised latents ──
    if model_type == 'diffusion_vae':
        print("\nEvaluating denoised latents at multiple timesteps ...")
        denoised_results = evaluate_diffusion_denoised(
            model, config, test_loader, device, num_clusters,
            args.covariance_type, output_dir, labels, supervised
        )

    # ── Save metrics.yaml ──
    metrics_path = output_dir / 'metrics.yaml'
    metrics_out = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in all_metrics.items()}
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics_out, f, default_flow_style=False, sort_keys=False)
    print(f"\nMetrics saved to {metrics_path}")

    # ── Save config snapshot ──
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # ── Summary ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Model:     {model_type}")
    print(f"  Dataset:   {dataset_name}")
    print(f"  Samples:   {z_data.shape[0]}")

    if supervised:
        print(f"  Accuracy:  {all_metrics['accuracy']:.4f}")
        print(f"  NMI:       {all_metrics['nmi']:.4f}")
        print(f"  ARI:       {all_metrics['ari']:.4f}")
    else:
        print(f"  Silhouette:        {all_metrics['silhouette']:.4f}")
        print(f"  Calinski-Harabasz: {all_metrics['calinski_harabasz']:.4f}")
        print(f"  Davies-Bouldin:    {all_metrics['davies_bouldin']:.4f}")

    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
