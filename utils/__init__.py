"""
Utility modules for VAE-GMM training framework.
"""

from .config import get_config, save_config
from .data_loader import get_data_loaders
from .metrics import (
    compute_metrics,
    cluster_accuracy,
    compute_unsupervised_metrics,
    reduce_dimensionality,
    save_metrics_report_yaml,
)
from .visualization import (
    plot_reconstructions,
    plot_latent_space,
    plot_training_curves,
    plot_training_curves_separate,
)
from .logger import MetricsLogger, TqdmLogger, setup_results_dir
from .atlas_visualization import (
    compute_gmm_probabilities_generic,
    collect_all_images,
    save_cluster_sample_grids_generic,
    build_and_save_atlas_A,
    build_and_save_atlas_laplacian,
    build_and_save_atlas_geodesic,
    save_gmm_means_atlas,
    compute_and_save_cluster_purity,
)

__all__ = [
    'get_config',
    'save_config',
    'get_data_loaders',
    'compute_metrics',
    'cluster_accuracy',
    'compute_unsupervised_metrics',
    'reduce_dimensionality',
    'save_metrics_report_yaml',
    'plot_reconstructions',
    'plot_latent_space',
    'plot_training_curves',
    'plot_training_curves_separate',
    'MetricsLogger',
    'TqdmLogger',
    'setup_results_dir',
    'compute_gmm_probabilities_generic',
    'collect_all_images',
    'save_cluster_sample_grids_generic',
    'build_and_save_atlas_A',
    'build_and_save_atlas_laplacian',
    'build_and_save_atlas_geodesic',
    'save_gmm_means_atlas',
    'compute_and_save_cluster_purity',
]
