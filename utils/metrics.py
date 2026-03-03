"""
Clustering evaluation metrics (supervised and unsupervised).
"""

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Tuple
from datetime import datetime


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate clustering accuracy using Hungarian algorithm.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted cluster assignments (N,)

    Returns:
        accuracy: Clustering accuracy in [0, 1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    accuracy = w[row_ind, col_ind].sum() / y_pred.size

    return accuracy


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute supervised clustering metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted cluster assignments (N,)

    Returns:
        dict with keys: accuracy, nmi, ari
    """
    acc = cluster_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return {
        'accuracy': acc,
        'nmi': nmi,
        'ari': ari
    }


def compute_unsupervised_metrics(z_data: np.ndarray, cluster_assignments: np.ndarray) -> dict:
    """
    Compute unsupervised clustering quality metrics (no ground truth needed).

    Args:
        z_data: Latent space representations (N, D).
        cluster_assignments: Predicted cluster labels (N,).

    Returns:
        dict with keys:
            - silhouette: Silhouette score in [-1, 1] (higher is better).
                Measures how similar samples are to their own cluster vs. nearest neighbor cluster.
            - calinski_harabasz: Calinski-Harabasz index (higher is better).
                Ratio of between-cluster dispersion to within-cluster dispersion.
            - davies_bouldin: Davies-Bouldin index (lower is better).
                Average similarity between each cluster and its most similar cluster.
    """
    # Need at least 2 clusters for these metrics
    n_unique = len(np.unique(cluster_assignments))
    if n_unique < 2:
        return {
            'silhouette': float('nan'),
            'calinski_harabasz': float('nan'),
            'davies_bouldin': float('nan'),
        }

    sil = silhouette_score(z_data, cluster_assignments)
    ch = calinski_harabasz_score(z_data, cluster_assignments)
    db = davies_bouldin_score(z_data, cluster_assignments)

    return {
        'silhouette': float(sil),
        'calinski_harabasz': float(ch),
        'davies_bouldin': float(db),
    }


def reduce_dimensionality(z: np.ndarray, method: str = 'tsne',
                          n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """
    Reduce latent space dimensionality for visualization.

    Args:
        z: Latent codes (N, D)
        method: 'tsne' or 'pca'
        n_components: Number of dimensions (usually 2)
        random_state: Random seed

    Returns:
        z_reduced: Reduced latent codes (N, n_components)
    """
    if method == 'tsne':
        # Adjust perplexity for small datasets
        n_samples = z.shape[0]
        perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity < n_samples
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    z_reduced = reducer.fit_transform(z)
    return z_reduced


def save_metrics_report_yaml(save_path: str, best_epoch: int, best_metrics: dict,
                             final_epoch: int, final_metrics: dict, config: dict):
    """
    Save comprehensive YAML report with best and final metrics.

    Args:
        save_path: Path to save YAML file (e.g., 'results/metrics_report.yaml')
        best_epoch: Epoch number of best checkpoint
        best_metrics: Dict of metrics at best epoch
        final_epoch: Final epoch number
        final_metrics: Dict of metrics at final epoch
        config: Full experiment configuration
    """
    # Extract experiment info
    experiment_info = {
        'name': f"{config['model']['name']}_{config['dataset']['name']}",
        'model': config['model']['name'],
        'dataset': config['dataset']['name'],
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # Build best checkpoint section
    best_checkpoint = {'epoch': best_epoch}
    for key, value in best_metrics.items():
        if isinstance(value, (int, float, np.number)):
            best_checkpoint[key] = float(value)

    # Build final checkpoint section
    final_checkpoint = {'epoch': final_epoch}
    for key, value in final_metrics.items():
        if isinstance(value, (int, float, np.number)):
            final_checkpoint[key] = float(value)

    # Calculate improvement deltas for key metrics
    improvement_keys = ['test_accuracy', 'test_nmi', 'test_ari', 'accuracy', 'nmi', 'ari']
    improvement = {}
    for key in improvement_keys:
        if key in best_metrics and key in final_metrics:
            delta = float(final_metrics[key]) - float(best_metrics[key])
            improvement[key] = round(delta, 6)

    # Determine best metric (prioritize test_accuracy, then accuracy)
    best_metric_key = None
    best_metric_value = None
    if 'test_accuracy' in best_metrics:
        best_metric_key = 'test_accuracy'
        best_metric_value = float(best_metrics['test_accuracy'])
    elif 'accuracy' in best_metrics:
        best_metric_key = 'accuracy'
        best_metric_value = float(best_metrics['accuracy'])

    summary = {
        'best_metric': best_metric_key,
        'best_metric_value': best_metric_value,
        'best_metric_epoch': best_epoch
    }

    if improvement:
        summary['improvement_best_to_final'] = improvement

    # Construct final report
    report = {
        'experiment': experiment_info,
        'best_checkpoint': best_checkpoint,
        'final_checkpoint': final_checkpoint,
        'summary': summary
    }

    # Save to YAML
    with open(save_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False, indent=2)
