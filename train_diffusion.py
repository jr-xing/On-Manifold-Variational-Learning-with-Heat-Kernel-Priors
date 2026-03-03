"""
Training script for Diffusion VAE-GMM models.

Two-phase training:
  Phase 1: Train VAE-GMM end-to-end (recon + KL + GMM losses)
  Phase 2: Freeze VAE, train latent-space denoiser (DDPM noise prediction)

Usage:
  # Full pipeline (Phase 1 + Phase 2)
  python train_diffusion.py --config configs/mnist/diffusion_vae_mnist.yaml

  # Skip Phase 1 with pretrained VAE checkpoint
  python train_diffusion.py --config configs/mnist/diffusion_vae_mnist.yaml \
      --vae_ckpt checkpoints/vae_gmm_mnist.pth

  # Override config settings via CLI
  python train_diffusion.py --config configs/mnist/diffusion_vae_mnist.yaml \
      --epochs 50 --iters 20000 --hidden_dim 256 --num_layers 5
"""

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import yaml
from pathlib import Path
from sklearn.mixture import GaussianMixture

from utils import (
    get_config, save_config, get_data_loaders, compute_metrics,
    plot_reconstructions, plot_latent_space, plot_training_curves,
    plot_training_curves_separate, save_metrics_report_yaml,
    MetricsLogger, TqdmLogger, setup_results_dir,
)
from models.diffusion_vae import get_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_diffusion_args():
    """Parse diffusion-specific CLI arguments not handled by get_config()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--vae_ckpt', type=str, default=None,
                        help='Path to pretrained VAE checkpoint (skips Phase 1)')
    parser.add_argument('--iters', type=int, default=None,
                        help='Denoiser training iterations (overrides config)')
    parser.add_argument('--max_timestep', type=int, default=None,
                        help='Max timestep for denoiser training')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Denoiser hidden dimension')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Denoiser number of layers')
    parser.add_argument('--n_mc', type=int, default=None,
                        help='Number of Monte Carlo samples for evaluation')
    extra_args, _ = parser.parse_known_args()
    return extra_args


# --- Phase 1: VAE Training ---------------------------------------------------


def train_vae_epoch(model, data_loader, optimizer, device, config, epoch, logger):
    """Train VAE for one epoch (Phase 1)."""
    model.train()
    epoch_metrics = {
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'gmm_loss': 0.0,
        'total_loss': 0.0,
    }

    loss_weights = config['loss_weights'].copy()

    # KL annealing
    if 'kl_annealing' in config:
        kl_start = config['kl_annealing'].get('start', 0.0)
        kl_end = config['kl_annealing'].get('end', loss_weights['kl'])
        anneal_epochs = config['kl_annealing'].get('epochs', 30)
        if epoch <= anneal_epochs:
            loss_weights['kl'] = kl_start + (kl_end - kl_start) * (epoch / anneal_epochs)
        else:
            loss_weights['kl'] = kl_end

    logger.log_message(f"Epoch {epoch} weights: recon={loss_weights['reconstruction']:.2f}, "
                       f"kl={loss_weights['kl']:.3f}, gmm={loss_weights['gmm']:.2f}")

    pbar = TqdmLogger(
        total=len(data_loader),
        desc=f"[Phase 1] Epoch {epoch}",
        use_tqdm=config['logging']['use_tqdm']
    )

    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon, mu, logvar, z = model(data)
        loss_dict = model.compute_loss(data, recon, mu, logvar, z, loss_weights)
        loss_dict['total_loss'].backward()

        if 'grad_clip_max_norm' in config.get('training', {}):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['grad_clip_max_norm']
            )

        optimizer.step()

        for key in epoch_metrics:
            if key in loss_dict:
                if key == 'total_loss':
                    epoch_metrics[key] += loss_dict[key].item()
                else:
                    epoch_metrics[key] += loss_dict[key]

        if (batch_idx + 1) % config['logging']['log_interval'] == 0:
            pbar.set_postfix({'loss': f"{loss_dict['total_loss'].item():.4f}"})
        pbar.update(1)

    pbar.close()

    n_batches = len(data_loader)
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    return epoch_metrics


def evaluate_clustering(model, data_loader, device, config):
    """Evaluate clustering performance."""
    model.eval()
    z_data, labels = model.extract_latent_features(data_loader, device)

    covariance_type = config.get('training', {}).get('eval_gmm_covariance', 'diag')
    n_init = config.get('training', {}).get('eval_gmm_n_init', 10)
    result = model.fit_gmm_and_evaluate(
        z_data, labels,
        covariance_type=covariance_type,
        n_init=n_init
    )

    metrics = compute_metrics(labels, result['gmm_labels'])

    return {
        'accuracy': metrics['accuracy'],
        'nmi': metrics['nmi'],
        'ari': metrics['ari'],
        'z_data': z_data,
        'labels': labels,
        'gmm': result['gmm'],
    }


def run_phase1(model, train_loader, test_loader, device, config, results_dir, logger, dataset_info):
    """Run Phase 1: VAE-GMM training."""
    print("\n" + "=" * 70)
    print("PHASE 1: VAE-GMM Training")
    print("=" * 70)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )

    epochs = config['training']['epochs']
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_vae_epoch(
            model, train_loader, optimizer, device, config, epoch, logger
        )

        eval_results = evaluate_clustering(model, train_loader, device, config)
        all_metrics = {**train_metrics, **{k: eval_results[k] for k in ['accuracy', 'nmi', 'ari']}}
        logger.log_metrics(epoch, all_metrics)

        log_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {all_metrics['total_loss']:.4f} "
            f"(Recon: {all_metrics['recon_loss']:.4f}, "
            f"KL: {all_metrics['kl_loss']:.4f}, "
            f"GMM: {all_metrics['gmm_loss']:.4f}) | "
            f"Acc: {all_metrics['accuracy']:.4f}, "
            f"NMI: {all_metrics['nmi']:.4f}"
        )
        logger.log_message(log_msg)

        # Save visualizations periodically
        if epoch % config['logging']['visualization_interval'] == 0 or epoch == 1:
            data_iter = iter(train_loader)
            sample_batch, _ = next(data_iter)
            sample_batch = sample_batch.to(device)
            with torch.no_grad():
                recon, _, _, _ = model(sample_batch)

            recon_path = results_dir / 'visualizations' / f'reconstructions_epoch_{epoch}.png'
            plot_reconstructions(sample_batch, recon, str(recon_path), n_samples=10)

            latent_path = results_dir / 'visualizations' / f'latent_space_epoch_{epoch}.png'
            plot_latent_space(
                eval_results['z_data'], eval_results['labels'],
                dataset_info['class_names'], str(latent_path), method='tsne'
            )

        # Save best model
        if all_metrics['accuracy'] > best_accuracy:
            best_accuracy = all_metrics['accuracy']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'config': config,
            }, results_dir / 'checkpoint_best.pth')
            logger.log_message(f"  -> Saved best model (acc: {best_accuracy:.4f})")

    # Save final VAE checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, results_dir / 'checkpoint_vae_final.pth')

    # Evaluate on test set
    test_eval = evaluate_clustering(model, test_loader, device, config)
    print(f"\nPhase 1 complete. Best train acc: {best_accuracy:.4f} (epoch {best_epoch})")
    print(f"Test set: Acc={test_eval['accuracy']:.4f}, NMI={test_eval['nmi']:.4f}, ARI={test_eval['ari']:.4f}")

    # Plot training curves
    plot_training_curves(logger.history, str(results_dir / 'training_curves_phase1.png'))
    plot_training_curves_separate(
        logger.history, str(results_dir / 'training_curves_phase1_separate.png'),
        use_log_scale=False
    )

    return best_accuracy, test_eval


# --- Phase 2: Denoiser Training ----------------------------------------------


def extract_latent_codes(model, data_loader, device):
    """Extract latent codes from frozen VAE encoder."""
    model.eval()
    z_list, y_list = [], []
    with torch.no_grad():
        for imgs, labels in data_loader:
            mu, _ = model.encode(imgs.to(device))
            z_list.append(mu)
            y_list.append(labels)
    z = torch.cat(z_list, 0)
    y = torch.cat(y_list, 0).cpu().numpy()
    return z, y


def gmm_eval(z_data, y_true, num_clusters, cov='diag', seed=42):
    """Evaluate clustering with GMM (with lexsort for determinism)."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    sort_idx = np.lexsort(z_data.T[::-1])
    z_data, y_true = z_data[sort_idx], y_true[sort_idx]

    gmm = GaussianMixture(n_components=num_clusters, covariance_type=cov, random_state=seed)
    preds = gmm.fit_predict(z_data)

    K = max(y_true.max(), preds.max()) + 1
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, preds):
        cm[t, p] += 1
    row, col = linear_sum_assignment(-cm)
    acc = cm[row, col].sum() / len(y_true)

    nmi = normalized_mutual_info_score(y_true, preds)
    ari = adjusted_rand_score(y_true, preds)

    return acc, nmi, ari


def run_phase2(model, train_loader, test_loader, device, config, results_dir, logger):
    """Run Phase 2: Denoiser training on frozen VAE latent codes."""
    print("\n" + "=" * 70)
    print("PHASE 2: Denoiser Training")
    print("=" * 70)

    diffusion_config = config.get('diffusion', {})
    num_clusters = config['model']['num_clusters']

    # Freeze VAE
    model.freeze_vae()
    model.eval()  # VAE in eval mode

    # Extract latent codes
    print("Extracting latent codes from frozen VAE...")
    z_train, y_train = extract_latent_codes(model, train_loader, device)
    z_test, y_test = extract_latent_codes(model, test_loader, device)
    print(f"  z_train: {z_train.shape}, z_test: {z_test.shape}")

    # Baseline evaluation
    acc_base_tr, nmi_base_tr, _ = gmm_eval(z_train.cpu().numpy(), y_train, num_clusters)
    acc_base_te, nmi_base_te, _ = gmm_eval(z_test.cpu().numpy(), y_test, num_clusters)
    print(f"\nBaseline (raw z): train_acc={acc_base_tr:.4f}, test_acc={acc_base_te:.4f}")

    # Denoiser training setup
    iters = diffusion_config.get('iters', 10000)
    max_timestep = diffusion_config.get('max_timestep', 15)
    denoiser_lr = diffusion_config.get('learning_rate', 1e-3)
    denoiser_batch_size = diffusion_config.get('batch_size', 256)
    n_mc = diffusion_config.get('n_mc', 50)
    eval_timesteps = diffusion_config.get('eval_timesteps', [1, 2, 3, 5, 7, 10, 14])

    # Only optimize denoiser parameters
    optimizer = optim.Adam(model.denoiser.parameters(), lr=denoiser_lr)

    n_params = sum(p.numel() for p in model.denoiser.parameters())
    print(f"\nDenoiser: {n_params:,} params "
          f"(hidden={diffusion_config.get('hidden_dim', 128)}, "
          f"layers={diffusion_config.get('num_layers', 3)})")
    print(f"Training: {iters} iters, max_t={max_timestep}, lr={denoiser_lr}")

    # Train denoiser
    model.denoiser.train()
    losses = []

    for it in range(iters):
        idx = torch.randint(0, z_train.shape[0], (denoiser_batch_size,), device=device)
        z_0 = z_train[idx]

        loss = model.compute_denoiser_loss(z_0, max_timestep)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (it + 1) % 2000 == 0 or it == 0:
            avg_loss = np.mean(losses[-200:])
            print(f"  Iter {it+1:6d}/{iters} | loss: {avg_loss:.6f}")

    final_loss = np.mean(losses[-200:])
    print(f"\nDenoiser training done. Final loss: {final_loss:.6f}")

    # Save denoiser checkpoint
    torch.save({
        'denoiser_state_dict': model.denoiser.state_dict(),
        'final_loss': final_loss,
        'iters': iters,
        'max_timestep': max_timestep,
        'hidden_dim': diffusion_config.get('hidden_dim', 128),
        'num_layers': diffusion_config.get('num_layers', 3),
    }, results_dir / 'denoiser.pth')

    # --- Evaluate ---
    model.denoiser.eval()

    print(f"\n{'=' * 80}")
    print(f"EVALUATION (n_mc={n_mc})")
    print(f"{'=' * 80}")
    print(f"\nBaseline: train_acc={acc_base_tr:.4f}  test_acc={acc_base_te:.4f}")
    print(f"\n  t | train_acc | test_acc | test_NMI | ||dz|| |")
    print(f"  --+----------+----------+----------+--------+")

    best_test_acc = 0
    best_test_t = 0
    results = {}

    for eval_t in eval_timesteps:
        torch.manual_seed(42)
        z_den_tr = model.onestep_denoise(z_train, eval_t, n_mc)
        torch.manual_seed(42)
        z_den_te = model.onestep_denoise(z_test, eval_t, n_mc)

        acc_tr, nmi_tr, ari_tr = gmm_eval(z_den_tr.cpu().numpy(), y_train, num_clusters)
        acc_te, nmi_te, ari_te = gmm_eval(z_den_te.cpu().numpy(), y_test, num_clusters)
        dist = float(torch.norm(z_den_te - z_test, dim=1).mean())

        marker = ' ***' if acc_te > acc_base_te else ''
        print(f"  {eval_t:2d} |  {acc_tr:.4f}  |  {acc_te:.4f}  |  {nmi_te:.4f}  | {dist:.4f} |{marker}")

        results[f't{eval_t}'] = {
            'train_acc': float(acc_tr), 'test_acc': float(acc_te),
            'train_nmi': float(nmi_tr), 'test_nmi': float(nmi_te),
            'train_ari': float(ari_tr), 'test_ari': float(ari_te),
            'dist': float(dist),
        }
        if acc_te > best_test_acc:
            best_test_acc = acc_te
            best_test_t = eval_t

    print(f"\nBest test: t={best_test_t}, acc={best_test_acc:.4f}")
    print(f"Target > {acc_base_te:.4f}: {'PASS' if best_test_acc > acc_base_te else 'FAIL'}")
    print(f"Delta vs baseline: {best_test_acc - acc_base_te:+.4f}")

    # Save metrics
    metrics = {
        'config': {
            'iters': iters,
            'max_timestep': max_timestep,
            'hidden_dim': diffusion_config.get('hidden_dim', 128),
            'num_layers': diffusion_config.get('num_layers', 3),
            'lr': denoiser_lr,
            'n_mc': n_mc,
            'final_loss': float(final_loss),
        },
        'baseline': {
            'train_acc': float(acc_base_tr),
            'test_acc': float(acc_base_te),
        },
        'best_test': {
            'timestep': best_test_t,
            'acc': float(best_test_acc),
        },
        'per_timestep': results,
    }
    with open(results_dir / 'metrics_diffusion.yaml', 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)

    print(f"\nPhase 2 results saved to: {results_dir}")
    return metrics


# --- Main ---------------------------------------------------------------------


def main():
    """Main training function."""
    # Parse diffusion-specific CLI args
    diffusion_args = parse_diffusion_args()

    # Get main config (handles --config, --output_dir, --epochs, --batch_size, --lr, --seed, --device)
    config = get_config()

    # Apply diffusion-specific CLI overrides
    if 'diffusion' not in config:
        config['diffusion'] = {}
    diffusion_config = config['diffusion']
    if diffusion_args.iters is not None:
        diffusion_config['iters'] = diffusion_args.iters
    if diffusion_args.max_timestep is not None:
        diffusion_config['max_timestep'] = diffusion_args.max_timestep
    if diffusion_args.hidden_dim is not None:
        diffusion_config['hidden_dim'] = diffusion_args.hidden_dim
    if diffusion_args.num_layers is not None:
        diffusion_config['num_layers'] = diffusion_args.num_layers
    if diffusion_args.n_mc is not None:
        diffusion_config['n_mc'] = diffusion_args.n_mc

    # Set seed
    set_seed(config['experiment']['seed'])

    # Setup device
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Setup results directory (use output_dir if specified)
    output_dir = config['experiment'].get('results_dir', './results')
    experiment_name = f"diffusion_vae_{config['dataset']['name']}"
    results_dir = setup_results_dir(output_dir, experiment_name)
    (results_dir / 'visualizations').mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")
    save_config(config, str(results_dir / 'config.yaml'))

    # Load data
    print("\nLoading dataset...")
    train_loader, test_loader, dataset_info = get_data_loaders(config)

    # Create model
    print("\nCreating model...")
    model = get_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Setup logger
    metrics_keys = ['recon_loss', 'kl_loss', 'gmm_loss', 'total_loss', 'accuracy', 'nmi', 'ari']
    logger = MetricsLogger(str(results_dir), metrics_keys)

    logger.log_message("=" * 60)
    logger.log_message("Diffusion VAE-GMM Training")
    logger.log_message("=" * 60)
    logger.log_message(f"Dataset: {config['dataset']['name']}")
    logger.log_message(f"Latent dim: {config['model']['latent_dim']}")
    logger.log_message(f"Num clusters: {config['model']['num_clusters']}")
    logger.log_message("=" * 60)

    # --- Phase 1: VAE Training ---

    if diffusion_args.vae_ckpt:
        # Skip Phase 1: load pretrained VAE
        print(f"\nLoading pretrained VAE from: {diffusion_args.vae_ckpt}")
        ckpt = torch.load(diffusion_args.vae_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']

        # Load VAE weights (encoder, decoder, gmm_means, gmm_logvars)
        # The pretrained checkpoint may be from a GMMVAE (without denoiser),
        # so we do partial loading of matching keys.
        model_dict = model.state_dict()
        pretrained_vae_keys = {k: v for k, v in state_dict.items()
                               if k in model_dict and not k.startswith('denoiser.')}
        model_dict.update(pretrained_vae_keys)
        model.load_state_dict(model_dict)

        loaded_keys = list(pretrained_vae_keys.keys())
        print(f"  Loaded {len(loaded_keys)} VAE parameters from checkpoint")
        logger.log_message(f"Loaded pretrained VAE from: {diffusion_args.vae_ckpt}")
        logger.log_message("Phase 1 skipped.")

        # Quick evaluation of loaded VAE
        test_eval = evaluate_clustering(model, test_loader, device, config)
        print(f"  VAE test set: Acc={test_eval['accuracy']:.4f}, "
              f"NMI={test_eval['nmi']:.4f}, ARI={test_eval['ari']:.4f}")
    else:
        # Run Phase 1: VAE training
        _, test_eval = run_phase1(
            model, train_loader, test_loader, device, config, results_dir, logger, dataset_info
        )

    # --- Phase 2: Denoiser Training ---

    phase2_metrics = run_phase2(
        model, train_loader, test_loader, device, config, results_dir, logger
    )

    # --- Save final combined checkpoint ---

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'phase2_metrics': phase2_metrics,
    }, results_dir / 'checkpoint_final.pth')

    logger.log_message(f"\nAll done! Results saved to: {results_dir}")
    logger.close()

    print(f"\nAll done! Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
