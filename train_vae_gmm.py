"""
Training script for VAE-GMM models.

Supports MNIST, cardiac, and OASIS datasets with size-based encoder/decoder
selection. Includes KL annealing, staged training, GMM evaluation, and
covariance type comparison.

Usage:
    python train_vae_gmm.py --config configs/mnist/vae_gmm.yaml
    python train_vae_gmm.py --config configs/cardiac/vae_gmm.yaml
    python train_vae_gmm.py --config configs/oasis/vae_gmm.yaml

    # Quick test run
    python train_vae_gmm.py --config configs/mnist/vae_gmm.yaml --epochs 2

    # Override settings via CLI
    python train_vae_gmm.py --config configs/mnist/vae_gmm.yaml --epochs 50 --batch_size 2048 --lr 0.0005

    # Specify output directory
    python train_vae_gmm.py --config configs/mnist/vae_gmm.yaml --output_dir results/my_experiment
"""

import torch
import torch.optim as optim
import numpy as np
import random
from pathlib import Path

from utils.config import get_config, save_config
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics, save_metrics_report_yaml
from utils.visualization import (
    plot_reconstructions,
    plot_latent_space,
    plot_training_curves,
    plot_training_curves_separate,
)
from utils.logger import MetricsLogger, TqdmLogger, setup_results_dir
from utils.atlas_visualization import (
    compute_gmm_probabilities_generic,
    collect_all_images,
    save_cluster_sample_grids_generic,
    compute_and_save_cluster_purity,
)
from models.vae_gmm import get_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_annealed_weights(epoch, config, current_metrics=None):
    """
    Compute annealed loss weights based on epoch.

    Supports:
    - Legacy: KL annealing, GMM annealing
    - Staged training with adaptive monitoring

    Args:
        epoch: Current training epoch (1-indexed)
        config: Configuration dictionary
        current_metrics: Optional dict with current metrics (for adaptive monitoring)

    Returns:
        Dictionary of loss weights with annealing applied
    """
    # Staged training
    if 'staged_training' in config and config['staged_training'].get('enabled', False):
        if not hasattr(get_annealed_weights, 'scheduler'):
            from utils.staged_training import StagedTrainingScheduler
            get_annealed_weights.scheduler = StagedTrainingScheduler(config['staged_training'])

        return get_annealed_weights.scheduler.get_weights_for_epoch(epoch, current_metrics or {})

    # Legacy annealing
    weights = config['loss_weights'].copy()

    # KL annealing: gradually increase KL weight
    if 'kl_annealing' in config:
        kl_start = config['kl_annealing'].get('start', 0.0)
        kl_end = config['kl_annealing'].get('end', weights['kl'])
        anneal_epochs = config['kl_annealing'].get('epochs', 30)
        if epoch <= anneal_epochs:
            weights['kl'] = kl_start + (kl_end - kl_start) * (epoch / anneal_epochs)
        else:
            weights['kl'] = kl_end

    # GMM annealing: delay GMM loss and/or gradually increase
    if 'gmm_annealing' in config:
        gmm_start = config['gmm_annealing'].get('start', 0.0)
        gmm_end = config['gmm_annealing'].get('end', weights['gmm'])
        anneal_epochs = config['gmm_annealing'].get('epochs', 30)
        warmup_start = config['gmm_annealing'].get('warmup_start', 0)

        if epoch < warmup_start:
            weights['gmm'] = 0.0
        elif epoch < warmup_start + anneal_epochs:
            progress = (epoch - warmup_start) / anneal_epochs
            weights['gmm'] = gmm_start + (gmm_end - gmm_start) * progress
        else:
            weights['gmm'] = gmm_end

    return weights


def train_epoch(model, data_loader, optimizer, device, config, epoch, logger):
    """Train for one epoch."""
    model.train()
    epoch_metrics = {
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'gmm_loss': 0.0,
        'total_loss': 0.0,
    }

    # Get annealed weights for this epoch
    prev_metrics = getattr(train_epoch, 'prev_metrics', {})
    annealed_weights = get_annealed_weights(epoch, config, prev_metrics)

    # Log stage transitions (if using staged training)
    if 'staged_training' in config and config['staged_training'].get('enabled', False):
        if hasattr(get_annealed_weights, 'scheduler'):
            get_annealed_weights.scheduler.log_stage_transition(epoch, logger)

    # Log current weights
    logger.log_message(
        f"Epoch {epoch} weights: recon={annealed_weights['reconstruction']:.2f}, "
        f"kl={annealed_weights['kl']:.3f}, gmm={annealed_weights['gmm']:.2f}"
    )

    # Progress bar
    pbar = TqdmLogger(
        total=len(data_loader),
        desc=f"Epoch {epoch}",
        use_tqdm=config['logging']['use_tqdm'],
    )

    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon, mu, logvar, z = model(data)

        # Compute losses with annealed weights
        loss_dict = model.compute_loss(data, recon, mu, logvar, z, annealed_weights)

        # Backward pass
        loss_dict['total_loss'].backward()

        # Gradient clipping (if configured)
        if 'grad_clip_max_norm' in config['training']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['grad_clip_max_norm'],
            )

        optimizer.step()

        # Accumulate metrics
        for key in epoch_metrics:
            if key in loss_dict:
                if key == 'total_loss':
                    epoch_metrics[key] += loss_dict[key].item()
                else:
                    epoch_metrics[key] += loss_dict[key]

        # Update progress bar
        if (batch_idx + 1) % config['logging']['log_interval'] == 0:
            pbar.set_postfix({'loss': f"{loss_dict['total_loss'].item():.4f}"})
        pbar.update(1)

    pbar.close()

    # Average metrics
    n_batches = len(data_loader)
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    # Store metrics for next epoch's adaptive monitoring
    train_epoch.prev_metrics = epoch_metrics.copy()

    return epoch_metrics


def evaluate_clustering(model, data_loader, device, config):
    """Evaluate clustering performance."""
    model.eval()

    # Extract latent features
    z_data, labels = model.extract_latent_features(data_loader, device)

    # Fit GMM and get predictions
    covariance_type = config.get('training', {}).get('eval_gmm_covariance', 'diag')
    n_init = config.get('training', {}).get('eval_gmm_n_init', 10)
    result = model.fit_gmm_and_evaluate(
        z_data, labels,
        covariance_type=covariance_type,
        n_init=n_init,
    )

    # Compute clustering metrics
    metrics = compute_metrics(labels, result['gmm_labels'])

    return {
        'accuracy': metrics['accuracy'],
        'nmi': metrics['nmi'],
        'ari': metrics['ari'],
        'z_data': z_data,
        'labels': labels,
        'gmm': result['gmm'],
    }


def save_visualizations(model, data_loader, eval_results, results_dir, epoch, dataset_info, config):
    """Save reconstruction, latent space, and optional cluster visualizations."""
    device = next(model.parameters()).device
    model.eval()

    # Get a batch for reconstruction visualization
    data_iter = iter(data_loader)
    sample_batch, _ = next(data_iter)
    sample_batch = sample_batch.to(device)

    with torch.no_grad():
        recon, _, _, _ = model(sample_batch)

    # Save reconstructions
    recon_path = results_dir / 'visualizations' / f'reconstructions_epoch_{epoch}.png'
    plot_reconstructions(sample_batch, recon, str(recon_path), n_samples=10)

    # Save latent space visualization
    latent_path = results_dir / 'visualizations' / f'latent_space_epoch_{epoch}.png'
    plot_latent_space(
        eval_results['z_data'],
        eval_results['labels'],
        dataset_info['class_names'],
        str(latent_path),
        method='tsne',
    )

    # Optional cluster-based visualizations
    viz_config = config.get('visualization', {})
    gmm = eval_results.get('gmm')
    z_data = eval_results['z_data']
    labels = eval_results['labels']

    if gmm is None:
        return

    # Compute cluster probabilities
    z_torch = torch.from_numpy(z_data).float().to(device)
    probs = compute_gmm_probabilities_generic(model, z_torch, device)

    # Cluster sample grids
    if viz_config.get('save_cluster_samples', False):
        X_all = collect_all_images(data_loader, device)
        save_cluster_sample_grids_generic(
            X_all, labels, probs,
            conf_thresh=viz_config.get('conf_thresh', 0.9),
            out_dir=results_dir / 'cluster_samples',
            epoch=epoch,
            class_names=dataset_info['class_names'],
            sample_topn=viz_config.get('sample_topn_per_cluster', 32),
            nrow=8,
        )

    # Cluster purity analysis
    if viz_config.get('save_purity', False):
        cluster_assignments = gmm.predict(z_data)
        compute_and_save_cluster_purity(
            cluster_assignments, labels,
            config['model']['num_clusters'],
            dataset_info['class_names'],
            results_dir,
            epoch,
        )


def main():
    """Main training function."""
    # Get configuration
    config = get_config()

    # Set random seed
    set_seed(config['experiment']['seed'])

    # Setup device
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Setup results directory
    resume_dir = config['training'].get('resume_dir', None)
    if resume_dir:
        results_dir = Path(resume_dir)
        print(f"\nResuming training from: {results_dir}")
    elif config['experiment'].get('results_dir') and config['experiment']['results_dir'] != './results':
        # --output_dir was provided (merged into results_dir by config system)
        results_dir = Path(config['experiment']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / 'visualizations').mkdir(exist_ok=True)
        print(f"\nResults will be saved to: {results_dir}")
        save_config(config, str(results_dir / 'config.yaml'))
    else:
        experiment_name = f"{config['model']['name']}_{config['dataset']['name']}"
        results_dir = setup_results_dir(
            config['experiment']['results_dir'],
            experiment_name,
        )
        print(f"\nResults will be saved to: {results_dir}")
        save_config(config, str(results_dir / 'config.yaml'))

    # Create output directories for optional visualizations
    viz_config = config.get('visualization', {})
    if viz_config.get('save_cluster_samples', False):
        (results_dir / 'cluster_samples').mkdir(exist_ok=True)

    # Load data
    print("\nLoading dataset...")
    train_loader, test_loader, dataset_info = get_data_loaders(config)

    # Create model
    print("\nCreating model...")
    model = get_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0),
    )

    # Setup logger
    metrics_keys = ['recon_loss', 'kl_loss', 'gmm_loss', 'total_loss', 'accuracy', 'nmi', 'ari']
    logger = MetricsLogger(str(results_dir), metrics_keys)

    logger.log_message("=" * 60)
    logger.log_message("Training Configuration")
    logger.log_message("=" * 60)
    logger.log_message(f"Model: {config['model']['name']}")
    logger.log_message(f"Dataset: {config['dataset']['name']}")
    logger.log_message(f"Epochs: {config['training']['epochs']}")
    logger.log_message(f"Batch size: {config['dataset']['batch_size']}")
    logger.log_message(f"Learning rate: {config['training']['learning_rate']}")
    logger.log_message(f"Latent dim: {config['model']['latent_dim']}")
    logger.log_message(f"Num clusters: {config['model']['num_clusters']}")
    logger.log_message("=" * 60)

    # Determine best checkpoint metric
    if config['dataset'].get('label_type') == 'none':
        best_metric = config['training'].get('best_checkpoint_metric', 'gmm_loss')
    else:
        best_metric = config['training'].get('best_checkpoint_metric', 'accuracy')

    maximize_metrics = ['accuracy', 'nmi', 'ari']
    minimize_metrics = ['gmm_loss', 'recon_loss', 'kl_loss', 'total_loss']

    if best_metric in maximize_metrics:
        best_value = 0.0
        is_better = lambda new, old: new > old
    elif best_metric in minimize_metrics:
        best_value = float('inf')
        is_better = lambda new, old: new < old
    else:
        raise ValueError(
            f"Unknown best_checkpoint_metric: {best_metric}. "
            f"Must be one of {maximize_metrics + minimize_metrics}"
        )

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_dir:
        checkpoint_path = results_dir / 'checkpoint_best.pth'
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck['epoch']
        saved_metric = ck.get('metric_value')
        if saved_metric is not None:
            best_value = saved_metric
        print(f"  Resumed from epoch {start_epoch} | {best_metric}={best_value:.4f}")

    # Training loop
    print("\nStarting training...")
    best_epoch = start_epoch
    best_metrics = {}
    logger.log_message(
        f"\n>>> Best checkpoint metric: {best_metric} "
        f"({'maximize' if best_metric in maximize_metrics else 'minimize'}) <<<\n"
    )
    if start_epoch > 0:
        logger.log_message(f">>> Resuming from epoch {start_epoch} <<<\n")

    for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, epoch, logger
        )

        # Evaluate clustering
        eval_results = evaluate_clustering(model, train_loader, device, config)

        # Combine metrics
        all_metrics = {
            **train_metrics,
            **{k: eval_results[k] for k in ['accuracy', 'nmi', 'ari']},
        }

        # Log metrics
        logger.log_metrics(epoch, all_metrics)

        # Print epoch summary
        log_msg = (
            f"Epoch {epoch}/{config['training']['epochs']} | "
            f"Loss: {all_metrics['total_loss']:.4f} "
            f"(Recon: {all_metrics['recon_loss']:.4f}, "
            f"KL: {all_metrics['kl_loss']:.4f}, "
            f"GMM: {all_metrics['gmm_loss']:.4f}) | "
            f"Acc: {all_metrics['accuracy']:.4f}, "
            f"NMI: {all_metrics['nmi']:.4f}, "
            f"ARI: {all_metrics['ari']:.4f}"
        )
        logger.log_message(log_msg)

        # Save visualizations periodically
        if epoch % config['logging']['visualization_interval'] == 0 or epoch == 1:
            save_visualizations(
                model, train_loader, eval_results, results_dir, epoch, dataset_info, config
            )

        # Save best model
        current_value = all_metrics[best_metric]
        if is_better(current_value, best_value):
            best_value = current_value
            best_epoch = epoch
            best_metrics = {
                **train_metrics,
                **{k: eval_results[k] for k in ['accuracy', 'nmi', 'ari']},
            }
            checkpoint_path = results_dir / 'checkpoint_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'best_value': best_value,
                'config': config,
            }, checkpoint_path)
            logger.log_message(f"  -> Saved best model ({best_metric}: {best_value:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    logger.log_message("\nFinal evaluation on test set...")
    test_eval_results = evaluate_clustering(model, test_loader, device, config)

    final_msg = (
        f"Test set results:\n"
        f"  Accuracy: {test_eval_results['accuracy']:.4f}\n"
        f"  NMI: {test_eval_results['nmi']:.4f}\n"
        f"  ARI: {test_eval_results['ari']:.4f}"
    )
    print(final_msg)
    logger.log_message(final_msg)

    # Save final visualizations
    print("\nSaving final visualizations...")
    save_visualizations(
        model, test_loader, test_eval_results, results_dir, 'final', dataset_info, config
    )

    # Plot training curves
    plot_training_curves(
        logger.history,
        str(results_dir / 'training_curves.png'),
    )
    plot_training_curves_separate(
        logger.history,
        str(results_dir / 'training_curves_separate.png'),
        use_log_scale=False,
    )
    plot_training_curves_separate(
        logger.history,
        str(results_dir / 'training_curves_separate_log.png'),
        use_log_scale=True,
    )

    # Load and visualize best checkpoint
    print("\nGenerating best checkpoint visualizations...")
    checkpoint_path = results_dir / 'checkpoint_best.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        best_eval_results = evaluate_clustering(model, test_loader, device, config)

        plot_latent_space(
            best_eval_results['z_data'],
            best_eval_results['labels'],
            dataset_info['class_names'],
            str(results_dir / 'latent_space_best.png'),
            method='tsne',
        )

        data_iter = iter(test_loader)
        sample_batch, _ = next(data_iter)
        sample_batch = sample_batch.to(device)
        with torch.no_grad():
            recon, _, _, _ = model(sample_batch)
        plot_reconstructions(
            sample_batch, recon,
            str(results_dir / 'reconstructions_best.png'),
            n_samples=10,
        )

    # Save final checkpoint
    final_checkpoint_path = results_dir / 'checkpoint_final.pth'
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_eval_results['accuracy'],
        'test_nmi': test_eval_results['nmi'],
        'test_ari': test_eval_results['ari'],
        'config': config,
    }, final_checkpoint_path)

    # === GMM Covariance Type Comparison ===
    covariance_type = config.get('training', {}).get('eval_gmm_covariance', 'diag')
    latent_dim = config['model']['latent_dim']
    max_dim_for_full = 1000

    if covariance_type == 'diag' and latent_dim <= max_dim_for_full:
        print("\n" + "=" * 70)
        print("GMM COVARIANCE TYPE COMPARISON")
        print("=" * 70)
        print(f"Training used: covariance_type='{covariance_type}'")
        print(f"  Test Accuracy: {test_eval_results['accuracy']:.4f}")
        print(f"  Test NMI: {test_eval_results['nmi']:.4f}")
        print(f"  Test ARI: {test_eval_results['ari']:.4f}")

        print("\nEvaluating with covariance_type='full' (slower but more accurate)...")
        z_data_test, labels_test = model.extract_latent_features(test_loader, device)
        full_result = model.fit_gmm_and_evaluate(
            z_data_test, labels_test,
            covariance_type='full',
        )
        full_metrics = compute_metrics(labels_test, full_result['gmm_labels'])

        print(f"With covariance_type='full':")
        print(f"  Test Accuracy: {full_metrics['accuracy']:.4f} "
              f"({full_metrics['accuracy'] - test_eval_results['accuracy']:+.4f})")
        print(f"  Test NMI: {full_metrics['nmi']:.4f} "
              f"({full_metrics['nmi'] - test_eval_results['nmi']:+.4f})")
        print(f"  Test ARI: {full_metrics['ari']:.4f} "
              f"({full_metrics['ari'] - test_eval_results['ari']:+.4f})")

        print("\nNote: 'full' covariance captures feature correlations but is slower.")
        print("To use 'full' by default, set training.eval_gmm_covariance='full' in config.")
        print("=" * 70 + "\n")
    elif covariance_type == 'diag' and latent_dim > max_dim_for_full:
        print("\n" + "=" * 70)
        print("GMM COVARIANCE TYPE COMPARISON")
        print("=" * 70)
        print(f"Training used: covariance_type='{covariance_type}'")
        print(f"  Test Accuracy: {test_eval_results['accuracy']:.4f}")
        print(f"  Test NMI: {test_eval_results['nmi']:.4f}")
        print(f"  Test ARI: {test_eval_results['ari']:.4f}")
        print(f"\nNote: Skipping 'full' covariance comparison (latent_dim={latent_dim} > {max_dim_for_full}).")
        print("Full covariance is computationally infeasible for high-dimensional spaces.")
        print("=" * 70 + "\n")

    # Save comprehensive metrics report
    print("\nSaving metrics report...")
    save_metrics_report_yaml(
        save_path=str(results_dir / 'metrics_report.yaml'),
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        final_epoch=config['training']['epochs'],
        final_metrics={
            'test_accuracy': test_eval_results['accuracy'],
            'test_nmi': test_eval_results['nmi'],
            'test_ari': test_eval_results['ari'],
            'recon_loss': train_metrics['recon_loss'],
            'kl_loss': train_metrics['kl_loss'],
            'gmm_loss': train_metrics['gmm_loss'],
            'total_loss': train_metrics['total_loss'],
        },
        config=config,
    )

    logger.log_message(f"\nResults saved to: {results_dir}")
    logger.close()

    print(f"\nAll done! Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
