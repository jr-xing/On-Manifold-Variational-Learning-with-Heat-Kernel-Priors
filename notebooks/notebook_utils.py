"""
Shared utilities for inference notebooks.

Provides model loading, inference, metrics, and visualization functions
used across all dataset notebooks (MNIST, Cardiac, OASIS).
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, get_default_config
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics, compute_unsupervised_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    'baseline_gmm': 'Baseline GMM',
    'vae_gmm': 'VAE-GMM',
    'diffusion_vae': 'Diffusion-VAE',
    'clast': 'Ours',
}

# Best denoising timesteps (from reference experiments)
BEST_DENOISE_T = {
    'mnist': 5,
    'cardiac': 2,
    'oasis': 5,
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _fill_defaults(config):
    """Fill missing config keys from defaults."""
    default = get_default_config()
    for key in default:
        if key not in config:
            config[key] = default[key]
        elif isinstance(default[key], dict):
            for subkey in default[key]:
                if subkey not in config[key]:
                    config[key][subkey] = default[key][subkey]
    return config


def _get_config(dataset, model_name):
    """Load and fill config for a dataset/model pair."""
    config_path = PROJECT_ROOT / 'configs' / dataset / f'{model_name}.yaml'
    config = load_yaml_config(str(config_path))
    config = _fill_defaults(config)
    # Resolve data root to absolute path relative to project root
    data_root = config['dataset'].get('root', './data')
    if not os.path.isabs(data_root):
        config['dataset']['root'] = str(PROJECT_ROOT / data_root)
    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _detect_model_type(model_name):
    """Map config model name to model type string."""
    if 'diffusion_vae' in model_name:
        return 'diffusion_vae'
    elif model_name in ('clast', 'ours'):
        return 'clast'
    elif 'vae_gmm' in model_name:
        return 'vae_gmm'
    elif 'baseline_gmm' in model_name:
        return 'baseline_gmm'
    elif 'baseline_kmeans' in model_name:
        return 'baseline_kmeans'
    raise ValueError(f"Unknown model: {model_name}")


def _is_neural(model_type):
    return model_type in ('vae_gmm', 'clast', 'diffusion_vae')


def load_model_and_config(dataset, model_name, device):
    """
    Load a pre-trained model and its config.

    Returns (model, config, model_type).
    """
    config = _get_config(dataset, model_name)
    model_type = _detect_model_type(model_name)

    # Import the right factory
    if model_type == 'vae_gmm':
        from models.vae_gmm import get_model
    elif model_type == 'clast':
        from models.ours import get_model
    elif model_type == 'diffusion_vae':
        from models.diffusion_vae import get_model
    elif model_type == 'baseline_gmm':
        from models.baseline_gmm import get_model
    elif model_type == 'baseline_kmeans':
        from models.baseline_kmeans import get_model

    model = get_model(config)

    # Load checkpoint(s)
    ckpt_dir = PROJECT_ROOT / 'checkpoints' / dataset / model_name
    is_baseline = model_type in ('baseline_gmm', 'baseline_kmeans')

    if not is_baseline:
        if model_type == 'diffusion_vae':
            # Two checkpoints: VAE + denoiser
            vae_path = ckpt_dir / 'checkpoint_vae.pth'
            den_path = ckpt_dir / 'denoiser.pth'
            ckpt_vae = torch.load(vae_path, map_location=device, weights_only=False)
            state = ckpt_vae.get('model_state_dict', ckpt_vae)
            model.load_state_dict(state, strict=False)  # VAE-only weights
            ckpt_den = torch.load(den_path, map_location=device, weights_only=False)
            den_state = ckpt_den.get('denoiser_state_dict', ckpt_den)
            model.denoiser.load_state_dict(den_state)
        else:
            # Check for finetuned checkpoint first (OASIS Ours)
            finetuned_path = ckpt_dir / 'checkpoint_finetuned.pth'
            best_path = ckpt_dir / 'checkpoint_best.pth'
            ckpt_path = finetuned_path if finetuned_path.exists() else best_path
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model, config, model_type


# ---------------------------------------------------------------------------
# Feature extraction & GMM fitting
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, model_type, data_loader, device, collect_images=True):
    """
    Extract features, labels, and (optionally) raw images.

    For neural models, features are lexicographically sorted to match
    the training evaluation protocol and ensure deterministic GMM fitting.

    Returns (z_data, labels, images).
    """
    if _is_neural(model_type):
        z_list, labels_list, img_list = [], [], []
        for batch, target in data_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            z_list.append(mu.cpu().numpy())
            labels_list.append(target.numpy())
            if collect_images:
                img_list.append(batch.cpu())
        z_data = np.vstack(z_list)
        labels = np.hstack(labels_list)
        images = torch.cat(img_list, dim=0) if collect_images else None
        # Lexicographic sort for deterministic GMM initialization —
        # must match the training evaluation protocol.
        idx = np.lexsort(z_data.T[::-1])
        z_data = z_data[idx]
        labels = labels[idx]
        if images is not None:
            images = images[idx]
        return z_data, labels, images
    else:
        z_data, labels = model.extract_latent_features(data_loader, device)
        images = None
        if collect_images:
            img_list = []
            for batch, _ in data_loader:
                img_list.append(batch)
            images = torch.cat(img_list, dim=0)
        return z_data, labels, images


def fit_gmm(z_data, num_clusters, covariance_type='full', n_init=10):
    """Fit sklearn GMM, return (gmm, cluster_labels)."""
    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=0,
        reg_covar=1e-4,
    )
    cluster_labels = gmm.fit_predict(z_data)
    return gmm, cluster_labels


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def run_inference(model, model_type, config, train_loader, test_loader, device,
                  denoise_t=None):
    """
    Full inference pipeline for one model.

    GMM is fit on **training** data and used to predict cluster labels on
    **test** data.  This matches the evaluation protocol used during training
    (see train_ours.py) and produces metrics consistent with the paper.

    Returns dict with keys:
        z_data, labels, images, gmm, cluster_labels, num_clusters, model_type
    """
    num_clusters = config['model']['num_clusters']
    # Use full covariance for neural models (low-dim latent space) —
    # captures feature correlations and gives significantly better accuracy.
    # Use diag for baselines (high-dim pixel space) where full is too slow.
    is_baseline = model_type in ('baseline_gmm', 'baseline_kmeans')
    covariance_type = 'diag' if is_baseline else 'full'

    if model_type == 'diffusion_vae' and denoise_t is not None:
        n_mc = config.get('diffusion', {}).get('n_mc', 50)
        # Denoised features for train set (GMM fitting)
        z_train, _ = model.extract_denoised_latent_features(
            train_loader, device, denoise_t=denoise_t, n_mc=n_mc
        )
        # Denoised features for test set (prediction)
        z_test, labels_test = model.extract_denoised_latent_features(
            test_loader, device, denoise_t=denoise_t, n_mc=n_mc
        )
        # Collect test images
        img_list = []
        for batch, _ in test_loader:
            img_list.append(batch)
        images = torch.cat(img_list, dim=0)
    elif is_baseline:
        # Baselines operate in high-dim pixel space — fitting on full
        # training set is too slow, so fit+predict on test data only.
        z_test, labels_test, images = extract_features(
            model, model_type, test_loader, device, collect_images=True
        )
        z_train = z_test
    else:
        # Neural models: fit GMM on training data, predict on test data
        z_train, _, _ = extract_features(
            model, model_type, train_loader, device, collect_images=False
        )
        z_test, labels_test, images = extract_features(
            model, model_type, test_loader, device, collect_images=True
        )

    # Fit GMM on train data (or test data for baselines), predict on test data
    n_init = 1 if is_baseline else 10
    gmm, _ = fit_gmm(z_train, num_clusters, covariance_type, n_init=n_init)
    cluster_labels = gmm.predict(z_test)

    return {
        'z_data': z_test,
        'labels': labels_test,
        'images': images,
        'gmm': gmm,
        'cluster_labels': cluster_labels,
        'num_clusters': num_clusters,
        'model_type': model_type,
        'model': model,
    }


def run_all_models(dataset, model_names, device, verbose=True):
    """
    Run inference for all models on a dataset.

    GMM is fit on training data and used to predict on test data.

    Returns dict: {model_name: result_dict}.
    Also returns (test_loader, dataset_info).
    """
    results = {}
    loaders = None

    for name in model_names:
        ckpt_dir = PROJECT_ROOT / 'checkpoints' / dataset / name
        if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
            if verbose:
                print(f"  Skipping {name}: no checkpoint found")
            continue

        if verbose:
            print(f"  Loading {MODEL_DISPLAY_NAMES.get(name, name)}...", end=' ')

        model, config, model_type = load_model_and_config(dataset, name, device)

        # Load data (reuse across models)
        if loaders is None:
            eval_config = dict(config)
            eval_config['dataset'] = dict(config['dataset'])
            eval_config['dataset']['shuffle'] = False
            eval_config['dataset']['num_workers'] = 0
            train_loader, test_loader, dataset_info = get_data_loaders(eval_config)
            loaders = (train_loader, test_loader, dataset_info)
        train_loader, test_loader, dataset_info = loaders

        denoise_t = BEST_DENOISE_T.get(dataset) if model_type == 'diffusion_vae' else None
        result = run_inference(model, model_type, config, train_loader, test_loader,
                               device, denoise_t)
        result['display_name'] = MODEL_DISPLAY_NAMES.get(name, name)
        results[name] = result

        if verbose:
            print("done")

    return results, (test_loader, dataset_info)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_metrics_table(results, supervised=True):
    """Build a pandas DataFrame comparing metrics across models."""
    rows = []
    for name, r in results.items():
        if supervised:
            m = compute_metrics(r['labels'], r['cluster_labels'])
            rows.append({
                'Model': r['display_name'],
                'Accuracy': f"{m['accuracy']:.4f}",
                'NMI': f"{m['nmi']:.4f}",
                'ARI': f"{m['ari']:.4f}",
            })
        else:
            m = compute_unsupervised_metrics(r['z_data'], r['cluster_labels'])
            rows.append({
                'Model': r['display_name'],
                'Silhouette': f"{m['silhouette']:.4f}",
                'Calinski-Harabasz': f"{m['calinski_harabasz']:.1f}",
                'Davies-Bouldin': f"{m['davies_bouldin']:.4f}",
            })
    df = pd.DataFrame(rows)
    df = df.set_index('Model')
    return df


# ---------------------------------------------------------------------------
# Cluster means visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_cluster_means(model, model_type, gmm, num_clusters, device):
    """
    Decode GMM cluster means into images.

    Returns numpy array of shape (K, H, W).
    """
    if _is_neural(model_type):
        means = torch.from_numpy(gmm.means_.astype(np.float32)).to(device)
        decoded = []
        for k in range(num_clusters):
            img = model.decode(means[k:k+1]).cpu().squeeze().numpy()
            decoded.append(img)
        return np.stack(decoded, axis=0)
    else:
        # Baseline: inverse-transform and reshape
        means_pixel = model.scaler.inverse_transform(gmm.means_)
        image_size = model.image_size
        return means_pixel.reshape(-1, image_size, image_size)


def plot_cluster_means_grid(results, model_order, device, title='Cluster Means Comparison',
                            cluster_order=None):
    """
    Plot a grid of cluster means: rows=models, cols=clusters.

    Args:
        results: dict from run_all_models
        model_order: list of model names in display order
        device: torch device
        title: figure title
        cluster_order: optional list of cluster indices for reordering columns
    """
    # Filter to available models
    available = [m for m in model_order if m in results]
    n_models = len(available)
    num_clusters = results[available[0]]['num_clusters']

    if cluster_order is None:
        cluster_order = list(range(num_clusters))

    fig, axes = plt.subplots(n_models, num_clusters,
                             figsize=(1.8 * num_clusters, 2.2 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, mname in enumerate(available):
        r = results[mname]
        means = decode_cluster_means(r['model'], r['model_type'], r['gmm'],
                                     num_clusters, device)
        for col, ki in enumerate(cluster_order):
            ax = axes[row, col]
            ax.imshow(np.clip(means[ki], 0, 1), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'C{ki}', fontsize=9)
        # Row label
        axes[row, 0].set_ylabel(r['display_name'], fontsize=10, rotation=0,
                                labelpad=70, va='center')

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Cluster samples visualization
# ---------------------------------------------------------------------------

def plot_cluster_samples(images, cluster_labels, cluster_idx, n_samples=16, ncols=4,
                         title=None):
    """Show sample images from a specific cluster."""
    mask = cluster_labels == cluster_idx
    indices = np.where(mask)[0]
    if len(indices) == 0:
        print(f"No samples in cluster {cluster_idx}")
        return

    # Take first n_samples
    indices = indices[:n_samples]
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(indices):
        r, c = divmod(i, ncols)
        img = images[idx].squeeze().numpy() if isinstance(images, torch.Tensor) else images[idx].squeeze()
        axes[r, c].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[r, c].axis('off')

    # Hide unused
    for i in range(len(indices), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')

    if title:
        plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_all_models_cluster_samples(results, model_order, cluster_idx, n_samples=8):
    """Show cluster samples side by side for all models."""
    available = [m for m in model_order if m in results]
    for mname in available:
        r = results[mname]
        print(f"\n{r['display_name']} — Cluster {cluster_idx} "
              f"({(r['cluster_labels'] == cluster_idx).sum()} samples)")
        plot_cluster_samples(r['images'], r['cluster_labels'], cluster_idx,
                             n_samples=n_samples, ncols=n_samples,
                             title=r['display_name'])


# ---------------------------------------------------------------------------
# Cluster purity (supervised datasets)
# ---------------------------------------------------------------------------

def compute_cluster_purity_table(labels, cluster_labels, num_clusters, class_names):
    """Compute per-cluster purity, return as DataFrame."""
    rows = []
    for k in range(num_clusters):
        mask = cluster_labels == k
        total = int(mask.sum())
        if total == 0:
            rows.append({'Cluster': k, 'Count': 0, 'Dominant': '-', 'Purity': 0.0})
            continue
        counter = Counter(labels[mask].tolist())
        dom_label, dom_count = counter.most_common(1)[0]
        row = {
            'Cluster': k,
            'Count': total,
            'Dominant': class_names[dom_label] if dom_label < len(class_names) else str(dom_label),
            'Purity': dom_count / total,
        }
        for ci, cn in enumerate(class_names):
            row[cn] = counter.get(ci, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OASIS ventricle size utilities
# ---------------------------------------------------------------------------

def load_ventricle_sizes():
    """Load ventricle bounding box areas from OASIS data."""
    bbox_path = PROJECT_ROOT / 'data' / 'oasis' / 'coronal_bboxes.npz'
    data = np.load(bbox_path, allow_pickle=True)
    bboxes = data['bboxes']  # (N, 4): [y, x, h, w] or similar
    # Area = width * height (last two columns)
    areas = bboxes[:, 2] * bboxes[:, 3]
    return areas, data['subject_ids']


def get_cluster_order_by_ventricle_size(cluster_labels, ventricle_sizes, num_clusters):
    """
    Return cluster indices ordered by ascending mean ventricle size.

    Args:
        cluster_labels: (N,) cluster assignments
        ventricle_sizes: (N,) ventricle areas per sample
        num_clusters: K

    Returns:
        ordered_indices: list of cluster indices from smallest to largest mean ventricle
        cluster_mean_sizes: dict mapping cluster_idx -> mean_size
    """
    # We need to align sizes with cluster_labels
    # cluster_labels may be shorter if only test set; sizes cover full dataset
    n = len(cluster_labels)
    sizes = ventricle_sizes[:n] if len(ventricle_sizes) >= n else ventricle_sizes

    mean_sizes = {}
    for k in range(num_clusters):
        mask = cluster_labels == k
        if mask.sum() > 0:
            mean_sizes[k] = float(sizes[mask].mean())
        else:
            mean_sizes[k] = 0.0

    ordered = sorted(mean_sizes, key=lambda k: mean_sizes[k])
    return ordered, mean_sizes


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def save_results_cache(results, dataset, cache_dir=None):
    """Save cluster assignments and key arrays to cache for quick sample browsing."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / 'cache'
    cache_dir.mkdir(exist_ok=True)

    for name, r in results.items():
        path = cache_dir / f'{dataset}_{name}.npz'
        save_dict = {
            'cluster_labels': r['cluster_labels'],
            'labels': r['labels'],
        }
        np.savez_compressed(path, **save_dict)


def load_results_cache(dataset, model_name, cache_dir=None):
    """Load cached cluster assignments."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / 'cache'
    path = cache_dir / f'{dataset}_{model_name}.npz'
    if path.exists():
        data = np.load(path)
        return dict(data)
    return None
