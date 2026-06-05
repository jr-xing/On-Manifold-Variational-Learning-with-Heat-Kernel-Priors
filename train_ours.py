"""
Training script for our manifold-aware atlas clustering model.

Our method requires unique per-epoch operations including GMM refitting,
conditional overwriting, and multiple types of atlas generation.

Usage:
    python train_ours.py --config configs/mnist/ours.yaml
    python train_ours.py --config configs/cardiac/ours.yaml --epochs 50
    python train_ours.py --config configs/cardiac/ours.yaml --pretrain_checkpoint checkpoints/cardiac/vae_gmm/checkpoint_best.pth
"""

import argparse
import torch
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from sklearn.mixture import GaussianMixture

# Import utilities
from utils import (
    get_config, save_config, get_data_loaders, compute_metrics,
    plot_latent_space, plot_training_curves, plot_training_curves_separate,
    save_metrics_report_yaml, MetricsLogger, TqdmLogger, setup_results_dir,
    plot_reconstructions
)
from utils.staged_training import StagedTrainingScheduler

# Import model factory
from models.ours import get_model

# Import atlas utilities
from models.ours.atlas_utils import (
    build_cluster_atlas_A,
    build_cluster_atlas_laplacian,
    build_cluster_atlas_geodesic_medoid,
    compute_cluster_purity,
    save_cluster_sample_grids,
    save_refit_gmm_means_atlas,
    save_vae_prior_means_atlas,
    save_atlas_A_images,
    save_atlas_laplacian_images,
    save_atlas_geodesic_images
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_ours_args():
    """Parse method-specific CLI arguments not handled by get_config()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to pretrained VAE checkpoint for transfer learning')
    extra_args, _ = parser.parse_known_args()
    return extra_args


def train_epoch(model, data_loader, optimizer, device, config, epoch, logger):
    """Train for one epoch with standard VAE forward-backward."""
    model.train()

    epoch_metrics = {
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'gmm_loss': 0.0,
        'total_loss': 0.0
    }

    # Progress bar
    pbar = TqdmLogger(
        total=len(data_loader),
        desc=f"Epoch {epoch}",
        use_tqdm=config['logging']['use_tqdm']
    )

    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon, mu, logvar, z = model(data)

        # Compute losses
        loss_dict = model.compute_loss(
            data, recon, mu, logvar, z,
            config['loss_weights']
        )

        # Backward pass
        loss_dict['total_loss'].backward()

        # Gradient clipping is important for stability with atlas updates.
        if 'grad_clip_max_norm' in config['training']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['grad_clip_max_norm']
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
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}"
            })
        pbar.update(1)

    pbar.close()

    # Average metrics
    n_batches = len(data_loader)
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    return epoch_metrics


def compute_gmm_probabilities(model, z_data_torch):
    """
    Compute GMM cluster probabilities from the model's trainable GMM prior.

    Args:
        model: VAE model with GMM prior
        z_data_torch: Latent codes [N, D] (torch tensor on device)

    Returns:
        probs: Cluster probabilities [N, K] (numpy array)
    """
    with torch.no_grad():
        # Compute log-likelihoods for each component
        diff = z_data_torch[:, None, :] - model.gmm_means[None, :, :]  # [N, K, D]
        var = torch.exp(model.gmm_logvars)[None, :, :] + 1e-6  # [1, K, D]
        logp = -0.5 * torch.sum(diff * diff / var + torch.log(var), dim=-1)  # [N, K]

        # Add log mixture weights
        log_pi = torch.nn.functional.log_softmax(model.logit_pi, dim=0)[None, :]  # [1, K]
        log_prob = log_pi + logp  # [N, K]

        # Normalize to get probabilities
        probs = torch.softmax(log_prob, dim=1)  # [N, K]

    return probs.cpu().numpy()


def ours_epoch_operations(model, train_loader, test_loader, device, config,
                           epoch, results_dir, dataset_info):
    """
    Method-specific per-epoch operations.

    Performs:
    1. Collect embeddings from train and test sets (single-pass, lexsorted)
    2. Refit sklearn GMM on train embeddings
    3. Evaluate clustering metrics
    4. Conditional GMM overwriting at overwrite_epoch
    5. Generate atlases (every atlas_interval epochs)
    6. Save cluster purity statistics
    7. Save cluster sample grids
    """
    model.eval()

    # 1. Collect embeddings and images in single pass (prevents shuffle mismatch)
    z_train_list, y_train_list, X_train_list = [], [], []

    with torch.no_grad():
        for batch, target in train_loader:
            batch_device = batch.to(device)
            mu, _ = model.encode(batch_device)

            z_train_list.append(mu.cpu())
            y_train_list.append(target)
            X_train_list.append(batch)  # Keep on CPU to save memory

    # Concatenate - all perfectly aligned
    z_train = torch.cat(z_train_list, dim=0).numpy()
    y_train = torch.cat(y_train_list, dim=0).numpy()
    X_train = torch.cat(X_train_list, dim=0)  # Single copy for both atlases & samples

    # Sort lexicographically to ensure deterministic ordering
    # (GMM k-means initialization depends on sample order in the array)
    sort_idx = np.lexsort(z_train.T[::-1])
    z_train = z_train[sort_idx]
    y_train = y_train[sort_idx]
    X_train = X_train[sort_idx]

    # Test set (no images needed for test set)
    z_test_list, y_test_list = [], []
    with torch.no_grad():
        for batch, target in test_loader:
            batch_device = batch.to(device)
            mu, _ = model.encode(batch_device)
            z_test_list.append(mu.cpu())
            y_test_list.append(target)

    z_test = torch.cat(z_test_list, dim=0).numpy()
    y_test = torch.cat(y_test_list, dim=0).numpy()

    # Sort test set for consistency
    sort_idx_test = np.lexsort(z_test.T[::-1])
    z_test = z_test[sort_idx_test]
    y_test = y_test[sort_idx_test]

    # 2. Refit GMM on train embeddings
    gmm = GaussianMixture(
        n_components=config['model']['num_clusters'],
        covariance_type=config['ours']['refit_covariance_type'],
        n_init=config['ours']['refit_n_init'],
        reg_covar=config['ours']['refit_reg_covar'],
        random_state=0
    )
    gmm.fit(z_train)

    # 3. Clustering evaluation
    train_pred = gmm.predict(z_train)
    test_pred = gmm.predict(z_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    # 4. GMM overwriting (conditional at specific epoch)
    if epoch == config['ours']['overwrite_epoch']:
        logger_msg = f"[Epoch {epoch}] Overwriting VAE prior with refitted GMM"
        print(f"\n{logger_msg}")
        model.overwrite_gmm_from_sklearn(
            gmm,
            ema_decay=config['ours']['overwrite_ema_decay']
        )

    # Compute GMM probabilities (needed for atlas generation and cluster samples)
    # Use refitted sklearn GMM for consistency with evaluation metrics
    probs = gmm.predict_proba(z_train)

    # 5. Atlas generation (every atlas_interval epochs)
    if epoch % config['ours']['atlas_interval'] == 0:
        print(f"[Epoch {epoch}] Generating atlases...")

        # Scheme-A atlas (pixel-space robust median/mean)
        if config['ours'].get('save_scheme_a_atlases', True):
            atlas_A = build_cluster_atlas_A(
                X_train.cpu(), probs,
                config['ours']['conf_thresh'],
                config['ours']['gamma_power'],
                config['ours']['atlas_robust'],
                config['ours']['sample_topn_per_cluster'],
                config['ours']['max_per_cluster_for_atlas']
            )
            save_atlas_A_images(atlas_A, results_dir / 'atlas_A', epoch)

        # Manifold atlas (method-dependent)
        if config['ours']['atlas_method'] == 'laplacian':
            atlas_manifold = build_cluster_atlas_laplacian(
                model, z_train, probs,
                config['ours']['conf_thresh'],
                config['ours']['gamma_power'],
                config['ours']['geodesic_knn_k'],
                config['ours']['geodesic_max_points_per_cluster']
            )
            save_atlas_laplacian_images(atlas_manifold, results_dir / 'atlas_laplacian', epoch)

        elif config['ours']['atlas_method'] == 'geodesic':
            atlas_manifold = build_cluster_atlas_geodesic_medoid(
                model, z_train, probs,
                config['ours']['conf_thresh'],
                config['ours']['gamma_power'],
                config['ours']['geodesic_knn_k'],
                config['ours']['geodesic_max_points_per_cluster'],
                seed=config['experiment']['seed'],
                epoch=epoch
            )
            save_atlas_geodesic_images(atlas_manifold, results_dir / 'atlas_geodesic', epoch)

        # Refit GMM means atlas
        if config['ours'].get('save_refit_gmm_atlases', True):
            save_refit_gmm_means_atlas(
                model, gmm,
                results_dir / 'atlas_refit_means',
                epoch, nrow=3
            )

        # VAE prior means atlas (disabled by default - training artifact)
        if config['ours'].get('save_vae_prior_atlases', False):
            save_vae_prior_means_atlas(
                model, results_dir / 'atlas_vae_prior', epoch, nrow=3
            )

    # 6. Cluster purity analysis (every epoch)
    purity_df = compute_cluster_purity(
        train_pred, y_train,
        config['model']['num_clusters'],
        dataset_info['class_names']
    )
    purity_dir = results_dir / 'cluster_purity'
    purity_dir.mkdir(exist_ok=True)
    purity_csv_path = purity_dir / f'cluster_purity_epoch_{epoch:03d}.csv'
    purity_df.to_csv(purity_csv_path, index=False)

    # 7. Cluster sample grids (every epoch)
    if config['ours'].get('save_cluster_samples', True):
        save_cluster_sample_grids(
            model, X_train, y_train, probs,
            config['ours']['conf_thresh'],
            results_dir / 'cluster_samples',
            epoch, dataset_info['class_names'],
            config['ours']['sample_topn_per_cluster'],
            nrow=8
        )

    # 8. Reconstruction plots (every visualization_interval epochs)
    visualization_interval = config['logging'].get('visualization_interval', 10)
    if epoch % visualization_interval == 0:
        # Get a sample batch for reconstruction visualization
        data_iter = iter(train_loader)
        sample_batch, _ = next(data_iter)
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            recon, _, _, _ = model(sample_batch)

        # Save reconstruction comparison
        recon_dir = results_dir / 'visualizations'
        recon_dir.mkdir(exist_ok=True, parents=True)
        recon_path = recon_dir / f'reconstructions_epoch_{epoch}.png'

        plot_reconstructions(sample_batch, recon, str(recon_path), n_samples=10)

        # Save latent space t-SNE
        latent_path = recon_dir / f'latent_space_epoch_{epoch}.png'

        plot_latent_space(
            z_train, y_train,
            dataset_info['class_names'],
            str(latent_path),
            method='tsne'
        )

    return {
        'train_accuracy': train_metrics['accuracy'],
        'train_nmi': train_metrics['nmi'],
        'train_ari': train_metrics['ari'],
        'test_accuracy': test_metrics['accuracy'],
        'test_nmi': test_metrics['nmi'],
        'test_ari': test_metrics['ari']
    }


def main():
    """Main training function."""
    # Parse method-specific CLI args
    ours_args = parse_ours_args()

    # Get configuration (handles --config, --output_dir, --epochs, --batch_size, --lr, --seed, --device)
    config = get_config()
    # Apply method-specific CLI overrides
    if ours_args.pretrain_checkpoint is not None:
        config['pretrain_checkpoint'] = ours_args.pretrain_checkpoint

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
    else:
        # Use output_dir if specified, otherwise auto-generate timestamped dir
        output_dir = config['experiment'].get('results_dir', './results')
        experiment_name = f"ours_{config['dataset']['name']}"
        results_dir = setup_results_dir(output_dir, experiment_name)
        print(f"\nResults will be saved to: {results_dir}")
        save_config(config, str(results_dir / 'config.yaml'))

    # Load data
    print("\nLoading dataset...")
    train_loader, test_loader, dataset_info = get_data_loaders(config)
    print(f"Classes: {dataset_info['num_classes']} ({', '.join(dataset_info['class_names'])})")

    # Create model
    print("\nCreating model...")
    model = get_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load pretrained checkpoint if provided
    if config.get('pretrain_checkpoint') is not None:
        print(f"\nLoading pretrained checkpoint from: {config['pretrain_checkpoint']}")
        checkpoint = torch.load(config['pretrain_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  -> Loaded model from epoch {checkpoint['epoch']}")
        if 'test_accuracy' in checkpoint:
            print(f"  -> Pretrained test accuracy: {checkpoint['test_accuracy']:.4f}")
        print("  -> Transfer learning enabled: Fine-tuning on new dataset")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )

    # Setup logger (append mode when resuming to preserve existing metrics)
    metrics_keys = [
        'recon_loss', 'kl_loss', 'gmm_loss', 'total_loss',
        'train_accuracy', 'train_nmi', 'train_ari',
        'test_accuracy', 'test_nmi', 'test_ari'
    ]
    logger = MetricsLogger(str(results_dir), metrics_keys, resume=bool(resume_dir))

    # Log configuration
    logger.log_message("=" * 60)
    logger.log_message("Ours Training Configuration")
    logger.log_message("=" * 60)
    logger.log_message(f"Dataset: {config['dataset']['name']}")
    logger.log_message(f"Atlas method: {config['ours']['atlas_method']}")
    logger.log_message(f"Epochs: {config['training']['epochs']}")
    logger.log_message(f"Batch size: {config['dataset']['batch_size']}")
    logger.log_message(f"Learning rate: {config['training']['learning_rate']}")
    logger.log_message(f"Latent dim: {config['model']['latent_dim']}")
    logger.log_message(f"Num clusters: {config['model']['num_clusters']}")
    logger.log_message(f"GMM overwrite epoch: {config['ours']['overwrite_epoch']}")

    # Log transfer learning / staged training configuration
    if config.get('pretrain_checkpoint') is not None:
        logger.log_message(f"Transfer learning: {config['pretrain_checkpoint']}")
    if config.get('vae_pretrain_epochs', 0) > 0:
        logger.log_message(f"Staged training: VAE-only for {config['vae_pretrain_epochs']} epochs, then joint")

    logger.log_message("=" * 60)

    # Create output directories
    output_subdirs = [
        'atlas_A', 'atlas_refit_means', 'atlas_vae_prior', 'cluster_samples'
    ]
    if config['ours']['atlas_method'] == 'laplacian':
        output_subdirs.append('atlas_laplacian')
    elif config['ours']['atlas_method'] == 'geodesic':
        output_subdirs.append('atlas_geodesic')

    for subdir in output_subdirs:
        (results_dir / subdir).mkdir(exist_ok=True)

    # Training loop
    print("\nStarting training...")

    # Determine best checkpoint metric
    # Auto-detect: use gmm_loss for unsupervised datasets (label_type='none')
    if config['dataset'].get('label_type') == 'none':
        best_metric = config['training'].get('best_checkpoint_metric', 'gmm_loss')
    else:
        best_metric = config['training'].get('best_checkpoint_metric', 'test_accuracy')

    # Determine if metric should be maximized or minimized
    maximize_metrics = ['test_accuracy', 'train_accuracy', 'test_nmi', 'test_ari', 'train_nmi', 'train_ari']
    minimize_metrics = ['gmm_loss', 'recon_loss', 'kl_loss', 'total_loss']

    if best_metric in maximize_metrics:
        best_value = 0.0
        is_better = lambda new, old: new > old
    elif best_metric in minimize_metrics:
        best_value = float('inf')
        is_better = lambda new, old: new < old
    else:
        raise ValueError(f"Unknown best_checkpoint_metric: {best_metric}. "
                        f"Must be one of {maximize_metrics + minimize_metrics}")

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_dir and not config.get('pretrain_checkpoint'):
        checkpoint_path = results_dir / 'checkpoint_best.pth'
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck['epoch']
        saved_metric = ck.get('best_value')
        if saved_metric is not None:
            best_value = saved_metric
        print(f"  Resumed from epoch {start_epoch} | {best_metric}={best_value:.4f}")

    best_epoch = start_epoch
    best_metrics = {}

    logger.log_message(f"\n>>> Best checkpoint metric: {best_metric} <<<")
    logger.log_message(f"    {'Maximizing' if best_metric in maximize_metrics else 'Minimizing'} {best_metric}\n")
    if start_epoch > 0:
        logger.log_message(f">>> Resuming from epoch {start_epoch} <<<\n")

    # Store original GMM weight for staged training
    original_gmm_weight = config['loss_weights']['gmm']
    vae_pretrain_epochs = config.get('vae_pretrain_epochs', 0)

    # Initialize staged training scheduler if enabled
    staged_scheduler = None
    if 'staged_training' in config and config['staged_training'].get('enabled', False):
        staged_scheduler = StagedTrainingScheduler(config['staged_training'])
        logger.log_message(f"\n>>> Advanced Staged Training Enabled (StagedTrainingScheduler) <<<")
        logger.log_message(f"    Stage 1 (VAE warmup): Epochs 1-{config['staged_training']['stage1_end_epoch']}")
        logger.log_message(f"    Stage 2 (Transition): Epochs {config['staged_training']['stage1_end_epoch']+1}-{config['staged_training']['stage2_end_epoch']}")
        logger.log_message(f"    Stage 3 (Adaptive): Epochs {config['staged_training']['stage2_end_epoch']+1}-{config['training']['epochs']}\n")

    for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
        # Advanced staged training (if enabled)
        if staged_scheduler is not None:
            prev_metrics = getattr(train_epoch, 'prev_metrics', {})
            annealed_weights = staged_scheduler.get_weights_for_epoch(epoch, prev_metrics)
            config['loss_weights'].update(annealed_weights)
            staged_scheduler.log_stage_transition(epoch, logger)
        else:
            # Fallback: Simple staging (backward compatible)
            if vae_pretrain_epochs > 0 and epoch <= vae_pretrain_epochs:
                config['loss_weights']['gmm'] = 0.0
                if epoch == 1:
                    logger.log_message(f"\n>>> VAE Pretraining Phase (Epochs 1-{vae_pretrain_epochs}): GMM loss disabled <<<\n")
            elif vae_pretrain_epochs > 0 and epoch == vae_pretrain_epochs + 1:
                config['loss_weights']['gmm'] = original_gmm_weight
                logger.log_message(f"\n>>> Joint Training Phase (Epochs {vae_pretrain_epochs+1}-{config['training']['epochs']}): GMM loss enabled (weight={original_gmm_weight}) <<<\n")

        # Standard training
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, epoch, logger
        )

        # Method-specific atlas/GMM operations
        ours_metrics = ours_epoch_operations(
            model, train_loader, test_loader, device, config,
            epoch, results_dir, dataset_info
        )

        # Combine metrics
        all_metrics = {**train_metrics, **ours_metrics}

        # Log metrics
        logger.log_metrics(epoch, all_metrics)

        # Print epoch summary
        log_msg = (
            f"Epoch {epoch}/{config['training']['epochs']} | "
            f"Loss: {all_metrics['total_loss']:.4f} "
            f"(R: {all_metrics['recon_loss']:.4f}, "
            f"KL: {all_metrics['kl_loss']:.4f}, "
            f"GMM: {all_metrics['gmm_loss']:.4f}) | "
            f"Train: Acc={all_metrics['train_accuracy']:.4f}, "
            f"NMI={all_metrics['train_nmi']:.4f}, "
            f"ARI={all_metrics['train_ari']:.4f} | "
            f"Test: Acc={all_metrics['test_accuracy']:.4f}, "
            f"NMI={all_metrics['test_nmi']:.4f}, "
            f"ARI={all_metrics['test_ari']:.4f}"
        )
        logger.log_message(log_msg)

        # Store metrics for next epoch (for adaptive monitoring)
        if staged_scheduler is not None:
            train_epoch.prev_metrics = all_metrics.copy()

        # Save best model based on configurable metric
        current_value = all_metrics[best_metric]
        if is_better(current_value, best_value):
            best_value = current_value
            best_epoch = epoch
            best_metrics = all_metrics.copy()
            checkpoint_path = results_dir / 'checkpoint_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'best_value': best_value,
                'config': config
            }, checkpoint_path)
            logger.log_message(f"  -> Saved best model ({best_metric}: {best_value:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    logger.log_message("\nFinal test set results:")
    logger.log_message(f"  Best {best_metric}: {best_value:.4f} (epoch {best_epoch})")

    # Final plots
    print("\nGenerating final visualizations...")
    plot_training_curves(
        logger.history,
        str(results_dir / 'training_curves.png')
    )

    plot_training_curves_separate(
        logger.history,
        str(results_dir / 'training_curves_separate.png'),
        use_log_scale=False
    )

    plot_training_curves_separate(
        logger.history,
        str(results_dir / 'training_curves_separate_log.png'),
        use_log_scale=True
    )

    # Final t-SNE visualization
    z_test, y_test = model.extract_latent_features(test_loader, device)
    plot_latent_space(
        z_test, y_test, dataset_info['class_names'],
        str(results_dir / 'latent_space_final.png'),
        method='tsne'
    )

    # Load and visualize best checkpoint
    print("\nGenerating best checkpoint visualizations...")
    checkpoint_path = results_dir / 'checkpoint_best.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract features from best model
        z_test_best, y_test_best = model.extract_latent_features(test_loader, device)

        # Save best checkpoint latent space
        plot_latent_space(
            z_test_best, y_test_best, dataset_info['class_names'],
            str(results_dir / 'latent_space_best.png'),
            method='tsne'
        )

        # Generate reconstruction plots for best checkpoint
        print("Generating final reconstruction visualizations...")
        data_iter = iter(test_loader)
        sample_batch, _ = next(data_iter)
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            recon, _, _, _ = model(sample_batch)

        recon_path = results_dir / 'reconstructions_best.png'
        plot_reconstructions(sample_batch, recon, str(recon_path), n_samples=10)

    # Save final checkpoint
    final_checkpoint_path = results_dir / 'checkpoint_final.pth'
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': all_metrics['test_accuracy'],
        'test_nmi': all_metrics['test_nmi'],
        'test_ari': all_metrics['test_ari'],
        'config': config
    }, final_checkpoint_path)

    # Save comprehensive metrics report
    print("\nSaving metrics report...")
    save_metrics_report_yaml(
        save_path=str(results_dir / 'metrics_report.yaml'),
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        final_epoch=config['training']['epochs'],
        final_metrics=all_metrics,
        config=config
    )

    logger.log_message(f"\nResults saved to: {results_dir}")
    logger.close()

    print(f"\nAll done! Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
