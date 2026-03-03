"""
Visualization utilities for training results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from .metrics import reduce_dimensionality


def plot_reconstructions(original: torch.Tensor, reconstructed: torch.Tensor,
                        save_path: str, n_samples: int = 10):
    """
    Plot original images alongside their reconstructions.

    Args:
        original: Original images (N, C, H, W)
        reconstructed: Reconstructed images (N, C, H, W)
        save_path: Path to save the plot
        n_samples: Number of samples to display
    """
    n_samples = min(n_samples, original.size(0))

    # Convert to numpy and move to CPU
    original = original[:n_samples].detach().cpu().numpy()
    reconstructed = reconstructed[:n_samples].detach().cpu().numpy()

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))

    for i in range(n_samples):
        # Original images (top row)
        axes[0, i].imshow(original[i, 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstructed images (bottom row)
        axes[1, i].imshow(reconstructed[i, 0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(z: np.ndarray, labels: np.ndarray, class_names: List[str],
                      save_path: str, method: str = 'tsne'):
    """
    Plot 2D projection of latent space.

    Args:
        z: Latent codes (N, D)
        labels: Ground truth labels (N,)
        class_names: List of class names
        save_path: Path to save the plot
        method: Dimensionality reduction method ('tsne' or 'pca')
    """
    # Reduce to 2D
    z_2d = reduce_dimensionality(z, method=method, n_components=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label_idx, color in zip(unique_labels, colors):
        mask = labels == label_idx
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                  c=[color], label=class_names[label_idx],
                  alpha=0.6, s=20)

    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title(f'Latent Space Visualization ({method.upper()})', fontsize=14)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(metrics_history: dict, save_path: str):
    """
    Plot training curves (losses and clustering metrics over epochs).

    Args:
        metrics_history: Dictionary with keys like 'recon_loss', 'kl_loss', etc.
                        Each value is a list of values per epoch
        save_path: Path to save the plot
    """
    # Determine which metrics are available
    loss_keys = ['recon_loss', 'kl_loss', 'gmm_loss', 'trajectory_loss', 'jacobian_loss', 'total_loss']
    metric_keys = ['accuracy', 'nmi', 'ari']

    available_losses = [k for k in loss_keys if k in metrics_history and len(metrics_history[k]) > 0]
    available_metrics = [k for k in metric_keys if k in metrics_history and len(metrics_history[k]) > 0]

    n_plots = (len(available_losses) > 0) + (len(available_metrics) > 0)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot losses
    if available_losses:
        ax = axes[plot_idx]
        for loss_key in available_losses:
            epochs = list(range(1, len(metrics_history[loss_key]) + 1))
            ax.plot(epochs, metrics_history[loss_key], label=loss_key, marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses', fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot clustering metrics
    if available_metrics:
        ax = axes[plot_idx]
        for metric_key in available_metrics:
            epochs = list(range(1, len(metrics_history[metric_key]) + 1))
            ax.plot(epochs, metrics_history[metric_key], label=metric_key.upper(), marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Clustering Metrics', fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves_separate(metrics_history: dict, save_path: str, use_log_scale: bool = False):
    """
    Plot training curves with separate subplots for each loss type.

    Args:
        metrics_history: Dictionary with keys like 'recon_loss', 'kl_loss', etc.
                        Each value is a list of values per epoch
        save_path: Path to save the plot
        use_log_scale: If True, use log scale for loss subplots (helpful for losses with different scales)
    """
    # Determine which metrics are available
    loss_keys = ['recon_loss', 'kl_loss', 'gmm_loss', 'trajectory_loss', 'jacobian_loss', 'total_loss']
    metric_keys = ['accuracy', 'nmi', 'ari']

    # Also check for train/test variants (used in CLAST)
    train_metric_keys = ['train_accuracy', 'train_nmi', 'train_ari']
    test_metric_keys = ['test_accuracy', 'test_nmi', 'test_ari']

    available_losses = [k for k in loss_keys if k in metrics_history and len(metrics_history[k]) > 0]
    available_metrics = [k for k in metric_keys if k in metrics_history and len(metrics_history[k]) > 0]
    available_train_metrics = [k for k in train_metric_keys if k in metrics_history and len(metrics_history[k]) > 0]
    available_test_metrics = [k for k in test_metric_keys if k in metrics_history and len(metrics_history[k]) > 0]

    # Total number of subplots needed
    n_loss_plots = len(available_losses)
    n_metric_plots = 0

    # Determine how many metric plots we need
    if available_metrics:
        n_metric_plots = 1  # Single plot for accuracy/nmi/ari
    elif available_train_metrics or available_test_metrics:
        n_metric_plots = 1  # Single plot for train/test metrics

    total_plots = n_loss_plots + n_metric_plots

    if total_plots == 0:
        return

    # Determine grid layout (use 3 columns for good balance)
    n_cols = min(3, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    # Flatten axes for easy indexing
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    plot_idx = 0

    # Plot each loss in its own subplot
    for loss_key in available_losses:
        ax = axes[plot_idx]
        epochs = list(range(1, len(metrics_history[loss_key]) + 1))
        ax.plot(epochs, metrics_history[loss_key], marker='o', markersize=4, linewidth=2, color='C0')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)

        # Format title: 'recon_loss' -> 'Recon Loss'
        title = loss_key.replace('_', ' ').title()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Apply log scale if requested
        if use_log_scale:
            ax.set_yscale('log')

        plot_idx += 1

    # Plot clustering metrics (combined in one subplot)
    if available_metrics:
        ax = axes[plot_idx]
        for metric_key in available_metrics:
            epochs = list(range(1, len(metrics_history[metric_key]) + 1))
            ax.plot(epochs, metrics_history[metric_key],
                   label=metric_key.upper(), marker='o', markersize=4, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Clustering Metrics', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        plot_idx += 1
    elif available_train_metrics or available_test_metrics:
        # CLAST-style train/test metrics
        ax = axes[plot_idx]

        # Plot train metrics
        for metric_key in available_train_metrics:
            epochs = list(range(1, len(metrics_history[metric_key]) + 1))
            metric_name = metric_key.replace('train_', '').upper()
            ax.plot(epochs, metrics_history[metric_key],
                   label=f'Train {metric_name}', marker='o', markersize=4, linewidth=2, linestyle='-')

        # Plot test metrics
        for metric_key in available_test_metrics:
            epochs = list(range(1, len(metrics_history[metric_key]) + 1))
            metric_name = metric_key.replace('test_', '').upper()
            ax.plot(epochs, metrics_history[metric_key],
                   label=f'Test {metric_name}', marker='s', markersize=4, linewidth=2, linestyle='--')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Clustering Metrics (Train & Test)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
