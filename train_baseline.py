"""
Training script for baseline (non-neural) clustering methods.

Fits sklearn GMM or K-means on flattened pixel space in a single pass,
then evaluates clustering metrics and generates visualizations.

Usage:
    python train_baseline.py --config configs/mnist/baseline_gmm.yaml
    python train_baseline.py --config configs/cardiac/baseline_kmeans.yaml
    python train_baseline.py --config configs/oasis/baseline_gmm.yaml

    # Specify output directory
    python train_baseline.py --config configs/mnist/baseline_gmm.yaml --output_dir results/my_baseline
"""

import time
import yaml
import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image

from utils.config import get_config, save_config
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics
from utils.logger import MetricsLogger, setup_results_dir
from utils.atlas_visualization import (
    compute_gmm_probabilities_generic,
    collect_all_images,
    save_cluster_sample_grids_generic,
    compute_and_save_cluster_purity,
)


def save_cluster_means(model, num_clusters, image_size, save_path):
    """
    Visualize cluster centroids in pixel space as images.

    Args:
        model: Baseline GMM or KMeans model with fitted sklearn model
        num_clusters: Number of clusters (K)
        image_size: Image dimension (28 for MNIST, 128 for cardiac/OASIS)
        save_path: Path to save the visualization
    """
    if hasattr(model, 'gmm') and model.gmm is not None:
        centroids = model.gmm.means_  # [K, D]
    elif hasattr(model, 'kmeans') and model.kmeans is not None:
        centroids = model.kmeans.cluster_centers_  # [K, D]
    else:
        raise ValueError("Model has no fitted GMM or KMeans")

    # Reshape to images: [K, D] -> [K, 1, H, W]
    centroids_images = centroids.reshape(num_clusters, 1, image_size, image_size)

    save_image(
        torch.from_numpy(centroids_images).float(),
        save_path,
        nrow=min(num_clusters, 5),
        normalize=True,
        padding=2,
        pad_value=1.0,
    )


def save_metrics_csv(train_metrics, test_metrics, save_path):
    """
    Save metrics to CSV (simplified for baseline methods).

    Only includes clustering metrics since baselines have no training losses.
    """
    import csv
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_accuracy', 'train_nmi', 'train_ari',
            'test_accuracy', 'test_nmi', 'test_ari',
        ])
        writer.writerow([
            1,
            train_metrics['accuracy'], train_metrics['nmi'], train_metrics['ari'],
            test_metrics['accuracy'], test_metrics['nmi'], test_metrics['ari'],
        ])


def save_checkpoint(model, test_metrics, config, save_path):
    """Save model checkpoint (minimal for baselines)."""
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_metrics['accuracy'],
        'test_nmi': test_metrics['nmi'],
        'test_ari': test_metrics['ari'],
        'config': config,
    }, save_path)


def save_metrics_report(train_metrics, test_metrics, save_path):
    """Save detailed metrics report to YAML."""
    report = {
        'training_metrics': {
            'accuracy': float(train_metrics['accuracy']),
            'nmi': float(train_metrics['nmi']),
            'ari': float(train_metrics['ari']),
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'nmi': float(test_metrics['nmi']),
            'ari': float(test_metrics['ari']),
        },
    }

    with open(save_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)


def train_baseline(config):
    """
    Main training function for baseline clustering methods.

    Steps:
    1. Setup (results dir, device, seed, logging)
    2. Load data
    3. Create model (baseline GMM or KMeans)
    4. Fit clustering model (single pass on training data)
    5. Evaluate on test set
    6. Generate cluster visualizations
    7. Save outputs
    """
    # 1. Setup
    print("=" * 70)
    print("BASELINE CLUSTERING TRAINING")
    print("=" * 70)

    # Setup results directory
    if config['experiment'].get('results_dir') and config['experiment']['results_dir'] != './results':
        # --output_dir was provided
        results_dir = Path(config['experiment']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_name = config['model']['name']
        dataset_name = config['dataset']['name']
        experiment_name = f"{model_name}_{dataset_name}"
        results_dir = setup_results_dir(
            config['experiment'].get('results_dir', './results'),
            experiment_name,
        )

    (results_dir / 'cluster_samples').mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    device = torch.device('cpu')  # Baselines don't need GPU
    print(f"Using device: {device}")

    # Set random seed
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup logger
    metrics_keys = ['train_accuracy', 'train_nmi', 'train_ari',
                    'test_accuracy', 'test_nmi', 'test_ari']
    logger = MetricsLogger(str(results_dir), metrics_keys)

    # 2. Load data
    logger.log_message("\nLoading dataset...")
    train_loader, test_loader, dataset_info = get_data_loaders(config)

    logger.log_message(f"Loaded {config['dataset']['name']} dataset:")
    logger.log_message(f"  - Training samples: {len(train_loader.dataset)}")
    logger.log_message(f"  - Test samples: {len(test_loader.dataset)}")
    logger.log_message(f"  - Num classes: {dataset_info['num_classes']}")
    logger.log_message(f"  - Image size: {dataset_info['image_size']}")
    logger.log_message(f"  - Batch size: {config['dataset']['batch_size']}")

    # 3. Create model
    logger.log_message("\nCreating model...")
    model_name = config['model']['name']

    if 'gmm' in model_name.lower():
        from models.baseline_gmm import get_model
        model_type = 'GMM'
    elif 'kmeans' in model_name.lower():
        from models.baseline_kmeans import get_model
        model_type = 'K-means'
    else:
        raise ValueError(f"Unknown baseline model type: {model_name}")

    model = get_model(config)
    logger.log_message(f"Model type: Baseline {model_type}")
    logger.log_message(f"Input dimension: {model.input_dim} (pixel space)")
    logger.log_message(f"Number of clusters: {config['model']['num_clusters']}")

    # 4. Fit clustering on training data
    logger.log_message("\n" + "=" * 70)
    logger.log_message("FITTING CLUSTERING MODEL")
    logger.log_message("=" * 70)

    start_time = time.time()

    # Extract features (flattened pixel space)
    logger.log_message("\nExtracting features from training data...")
    z_train, y_train = model.extract_latent_features(train_loader, device)
    logger.log_message(f"Feature shape: {z_train.shape}")

    # Fit clustering
    logger.log_message(f"\nFitting {model_type} on pixel space...")
    covariance_type = config.get('training', {}).get('eval_gmm_covariance', 'diag')
    if model_type == 'GMM':
        logger.log_message(f"Covariance type: {covariance_type}")

    result = model.fit_gmm_and_evaluate(z_train, y_train, covariance_type=covariance_type)

    # Compute training metrics
    train_metrics = compute_metrics(y_train, result['gmm_labels'])

    elapsed = time.time() - start_time
    logger.log_message(f"\nClustering completed in {elapsed:.2f}s")
    logger.log_message(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    logger.log_message(f"Training NMI: {train_metrics['nmi']:.4f}")
    logger.log_message(f"Training ARI: {train_metrics['ari']:.4f}")

    # 5. Evaluate on test set
    logger.log_message("\n" + "=" * 70)
    logger.log_message("EVALUATING ON TEST SET")
    logger.log_message("=" * 70)

    logger.log_message("\nExtracting features from test data...")
    z_test, y_test = model.extract_latent_features(test_loader, device)

    logger.log_message("Predicting cluster assignments...")
    test_labels = result['gmm'].predict(z_test)
    test_metrics = compute_metrics(y_test, test_labels)

    logger.log_message(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    logger.log_message(f"Test NMI: {test_metrics['nmi']:.4f}")
    logger.log_message(f"Test ARI: {test_metrics['ari']:.4f}")

    # 6. Generate visualizations
    logger.log_message("\n" + "=" * 70)
    logger.log_message("GENERATING VISUALIZATIONS")
    logger.log_message("=" * 70)

    # Collect all training images
    logger.log_message("\nCollecting training images...")
    X_train = collect_all_images(train_loader, device)
    logger.log_message(f"Collected {X_train.shape[0]} images")

    # Compute cluster probabilities
    logger.log_message("\nComputing cluster probabilities...")
    z_train_torch = torch.from_numpy(z_train).float().to(device)
    probs = compute_gmm_probabilities_generic(model, z_train_torch, device)

    # Cluster means
    logger.log_message("\nSaving cluster means visualization...")
    image_size = dataset_info['image_size'][1]  # H dimension
    save_cluster_means(
        model,
        config['model']['num_clusters'],
        image_size,
        results_dir / 'cluster_means.png',
    )
    logger.log_message("  Saved to: cluster_means.png")

    # Cluster samples
    logger.log_message("\nSaving cluster sample grids...")
    save_cluster_sample_grids_generic(
        X_train, y_train, probs,
        conf_thresh=0.9,
        out_dir=results_dir / 'cluster_samples',
        epoch='final',
        class_names=dataset_info.get('class_names', [str(i) for i in range(dataset_info['num_classes'])]),
        sample_topn=32,
        nrow=8,
    )
    logger.log_message("  Saved to: cluster_samples/")

    # Cluster purity (only if we have ground truth labels)
    if not np.all(y_train == -1):
        logger.log_message("\nComputing cluster purity...")
        cluster_assignments = result['gmm'].predict(z_train)
        compute_and_save_cluster_purity(
            cluster_assignments,
            y_train,
            config['model']['num_clusters'],
            dataset_info.get('class_names', [str(i) for i in range(dataset_info['num_classes'])]),
            results_dir,
            epoch='final',
        )
        logger.log_message("  Saved to: cluster_purity_final.csv")
    else:
        logger.log_message("\nSkipping cluster purity (unsupervised, no ground truth labels)")

    # 7. Save outputs
    logger.log_message("\n" + "=" * 70)
    logger.log_message("SAVING OUTPUTS")
    logger.log_message("=" * 70)

    logger.log_message("\nSaving configuration...")
    save_config(config, str(results_dir / 'config.yaml'))

    logger.log_message("Saving metrics...")
    save_metrics_csv(train_metrics, test_metrics, results_dir / 'metrics.csv')
    save_metrics_report(train_metrics, test_metrics, results_dir / 'metrics_report.yaml')

    logger.log_message("Saving checkpoint...")
    save_checkpoint(model, test_metrics, config, results_dir / 'checkpoint_final.pth')

    logger.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  NMI: {test_metrics['nmi']:.4f}")
    print(f"  ARI: {test_metrics['ari']:.4f}")
    print("\nGenerated visualizations:")
    print(f"  - cluster_means.png (cluster centroids)")
    print(f"  - cluster_samples/ (example images per cluster)")
    if not np.all(y_train == -1):
        print(f"  - cluster_purity_final.csv (cluster purity analysis)")
    print("=" * 70 + "\n")


def main():
    """Parse arguments and run baseline training."""
    config = get_config()

    # Verify this is a baseline model
    model_name = config['model']['name']
    if 'baseline' not in model_name.lower():
        print(f"Error: This script is for baseline models only.")
        print(f"Got model: {model_name}")
        print(f"Use train_vae_gmm.py for VAE-GMM models.")
        import sys
        sys.exit(1)

    train_baseline(config)


if __name__ == '__main__':
    main()
