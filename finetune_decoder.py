"""
Fine-tune decoder (encoder frozen) to improve reconstruction sharpness.

Rationale:
  Models trained with simultaneous KL + GMM losses often leave the decoder
  under-trained for reconstruction. Freezing the encoder preserves the learned
  latent clustering; fine-tuning the decoder makes decode(GMM.means_) produce
  sharper outputs (e.g., for paper figures).

Supports all neural model types: vae_gmm, ours, diffusion_vae.

Loss options:
  - mse:  Mean Squared Error
  - bce:  Binary Cross-Entropy
  - l1:   L1 (Mean Absolute Error)
  - ssim: Differentiable SSIM (1 - SSIM)

Usage:
  python finetune_decoder.py --config configs/oasis/ours.yaml \
      --checkpoint checkpoints/oasis/ours/checkpoint_best.pth \
      --output_dir results/oasis_finetuned/ \
      --loss ssim --epochs 200 --lr 5e-5

  python finetune_decoder.py --config configs/mnist/vae_gmm.yaml \
      --checkpoint checkpoints/mnist/vae_gmm/checkpoint_best.pth \
      --output_dir results/mnist_finetuned/ \
      --loss mse --epochs 100 --lr 1e-4
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

from utils.config import load_yaml_config, get_default_config
from utils.data_loader import get_data_loaders


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune decoder with encoder frozen',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save fine-tuning outputs')
    parser.add_argument('--loss', type=str, default='ssim',
                        choices=['mse', 'bce', 'l1', 'ssim'],
                        help='Reconstruction loss function')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--lr_end', type=float, default=None,
                        help='Final learning rate for cosine annealing (default: lr/10)')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g., cuda:0, cpu)')
    return parser.parse_args()


# =============================================================================
# Differentiable SSIM loss
# =============================================================================

def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    """1D Gaussian kernel."""
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x.pow(2.0) / (2.0 * sigma ** 2))
    return gauss / gauss.sum()


def _create_ssim_window(window_size: int, channels: int) -> torch.Tensor:
    """Create 2D SSIM convolution window from separable 1D Gaussian."""
    k1d = _gaussian_kernel_1d(window_size, 1.5).unsqueeze(1)
    k2d = k1d.mm(k1d.t()).unsqueeze(0).unsqueeze(0)           # [1, 1, W, W]
    return k2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor,
              window_size: int = 11,
              C1: float = 0.01 ** 2,
              C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    Differentiable SSIM loss: returns 1 - mean(SSIM).
    Minimizing this loss maximizes structural similarity.

    Args:
        img1: Predicted images [B, C, H, W] in [0, 1]
        img2: Target images [B, C, H, W] in [0, 1]
        window_size: Gaussian window size
        C1, C2: Stability constants

    Returns:
        Scalar loss in [0, 1]
    """
    channels = img1.shape[1]
    window = _create_ssim_window(window_size, channels).to(img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    s1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    s2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))

    return 1.0 - ssim_map.mean()


# =============================================================================
# Loss dispatcher
# =============================================================================

def compute_loss(recon: torch.Tensor, x: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Compute reconstruction loss."""
    if loss_type == 'mse':
        return F.mse_loss(recon, x)
    elif loss_type == 'bce':
        return F.binary_cross_entropy(recon, x)
    elif loss_type == 'l1':
        return F.l1_loss(recon, x)
    elif loss_type == 'ssim':
        return ssim_loss(recon, x)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
    else:
        raise ValueError(
            f"Decoder fine-tuning is only supported for neural models "
            f"(vae_gmm, ours, diffusion_vae). Got: {model_name}"
        )


def load_model_and_checkpoint(config, model_type, checkpoint_path, device):
    """Instantiate model and load checkpoint weights."""
    if model_type == 'vae_gmm':
        from models.vae_gmm import get_model
    elif model_type == 'ours':
        from models.ours import get_model
    elif model_type == 'diffusion_vae':
        from models.diffusion_vae import get_model
    else:
        raise ValueError(f"Unsupported model type for fine-tuning: {model_type}")

    model = get_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, ckpt


# =============================================================================
# Data collection and GMM means
# =============================================================================

@torch.no_grad()
def collect_latent_codes(model, data_loader, device):
    """
    Single-pass extraction of encoder mu, lexicographically sorted
    for deterministic GMM fitting.

    Returns:
        mu_sorted: (N, D) numpy array of latent means
    """
    model.eval()
    mu_list = []

    for batch, _ in data_loader:
        batch = batch.to(device)
        mu, _ = model.encode(batch)
        mu_list.append(mu.cpu())

    mu_all = torch.cat(mu_list, dim=0)
    mu_np = mu_all.numpy()

    sort_idx = np.lexsort(mu_np.T[::-1])
    return mu_np[sort_idx]


@torch.no_grad()
def decode_gmm_means(model, mu_all, num_clusters, device,
                     covariance_type='full', reg_covar=1e-4, n_init=10):
    """
    Fit GMM on latent codes and decode cluster means to image space.

    Returns:
        means_images: (K, H, W) numpy array of decoded cluster means
    """
    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=0,
    )
    gmm.fit(mu_all)

    means_tensor = torch.from_numpy(gmm.means_.astype(np.float32)).to(device)
    decoded = []
    for k in range(num_clusters):
        img = model.decode(means_tensor[k:k + 1]).cpu().squeeze().numpy()
        decoded.append(img)

    return np.stack(decoded, axis=0)  # (K, H, W)


# =============================================================================
# Visualization helpers
# =============================================================================

def save_means_grid(means, path, title=''):
    """Save a 1xK grid of cluster mean images."""
    K = means.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(2 * K, 2.5))
    if K == 1:
        axes = [axes]
    for ki, ax in enumerate(axes):
        ax.imshow(np.clip(means[ki], 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'C{ki}', fontsize=8)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_comparison_grid(before, after, path, loss_type):
    """Save a 2xK grid: top row = before fine-tuning, bottom row = after."""
    K = before.shape[0]
    fig, axes = plt.subplots(2, K, figsize=(2 * K, 5))
    if K == 1:
        axes = axes.reshape(2, 1)
    for ki in range(K):
        for row, (data, label) in enumerate([
            (before, 'Before'),
            (after, f'After ({loss_type})')
        ]):
            ax = axes[row, ki]
            ax.imshow(np.clip(data[ki], 0, 1), cmap='gray', vmin=0, vmax=1)
            if ki == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_title(f'C{ki}', fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t_start = time.time()

    # ── Output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──
    log_path = output_dir / 'training.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    # ── Load config ──
    config = load_yaml_config(args.config)
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
    num_clusters = config['model']['num_clusters']

    # Determine LR schedule
    lr_end = args.lr_end if args.lr_end is not None else args.lr / 10

    log("=" * 60)
    log("DECODER FINE-TUNING")
    log("=" * 60)
    log(f"  Model type:    {model_type}")
    log(f"  Dataset:       {config['dataset']['name']}")
    log(f"  Loss:          {args.loss}")
    log(f"  Epochs:        {args.epochs}")
    log(f"  LR:            {args.lr} -> {lr_end:.2e} (cosine)")
    log(f"  Checkpoint:    {args.checkpoint}")
    log(f"  Output:        {output_dir}")
    log(f"  Device:        {device}")
    log("=" * 60)

    # ── Load model ──
    log("\nLoading model ...")
    model, ckpt = load_model_and_checkpoint(config, model_type, args.checkpoint, device)
    log(f"  Loaded from epoch {ckpt.get('epoch', '?')}")

    # ── Freeze everything except decoder ──
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('decoder.')

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable params: {n_trainable:,} / {n_total:,} (decoder only)")

    # ── Collect latent codes (frozen encoder) for GMM ──
    log("\nCollecting latent codes for GMM fitting ...")
    # Use non-shuffled loader for deterministic codes
    collect_config = dict(config)
    collect_config['dataset'] = dict(config['dataset'])
    collect_config['dataset']['shuffle'] = False
    collect_config['dataset']['num_workers'] = 0
    collect_loader, _, _ = get_data_loaders(collect_config)

    model.eval()
    mu_all = collect_latent_codes(model, collect_loader, device)
    log(f"  {mu_all.shape[0]} samples, latent_dim={mu_all.shape[1]}")

    # ── GMM parameters for decoding means ──
    ours_config = config.get('ours', {})
    gmm_cov_type = ours_config.get('refit_covariance_type', 'full')
    gmm_reg = ours_config.get('refit_reg_covar', 1e-4)

    # ── Before: decode GMM means ──
    log("\nDecoding GMM means (before fine-tuning) ...")
    means_before = decode_gmm_means(
        model, mu_all, num_clusters, device,
        covariance_type=gmm_cov_type, reg_covar=gmm_reg
    )
    save_means_grid(means_before, output_dir / 'gmm_means_before.png', 'Before fine-tuning')
    np.save(output_dir / 'gmm_means_before.npy', means_before)
    log("  Saved gmm_means_before.png")

    # ── Training data loader (shuffled) ──
    train_config = dict(config)
    train_config['dataset'] = dict(config['dataset'])
    train_config['dataset']['shuffle'] = True
    train_loader, _, _ = get_data_loaders(train_config)

    # ── Optimizer (decoder parameters only) ──
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # ── Cosine annealing LR scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr_end
    )

    # ── Training loop ──
    log(f"\nStarting decoder fine-tuning ({args.epochs} epochs) ...")
    epoch_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for batch, _ in train_loader:
            batch = batch.to(device)

            # Encoder is frozen: no gradients needed
            with torch.no_grad():
                mu, _ = model.encode(batch)

            recon = model.decode(mu)
            loss = compute_loss(recon, batch, args.loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        scheduler.step()
        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t_start
        log(f"  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.6f}"
            f"  lr={current_lr:.2e}  elapsed={elapsed:.0f}s")

        # Intermediate visualization every 25 epochs or at end
        if epoch % 25 == 0:
            model.eval()
            means_mid = decode_gmm_means(
                model, mu_all, num_clusters, device,
                covariance_type=gmm_cov_type, reg_covar=gmm_reg
            )
            save_means_grid(
                means_mid, output_dir / f'gmm_means_epoch_{epoch:03d}.png',
                f'Epoch {epoch} ({args.loss})'
            )

    # ── After: decode GMM means ──
    log("\nDecoding GMM means (after fine-tuning) ...")
    model.eval()
    means_after = decode_gmm_means(
        model, mu_all, num_clusters, device,
        covariance_type=gmm_cov_type, reg_covar=gmm_reg
    )
    save_means_grid(means_after, output_dir / 'gmm_means_after.png',
                    f'After fine-tuning ({args.loss})')
    np.save(output_dir / 'gmm_means_after.npy', means_after)

    # Before/after comparison
    save_comparison_grid(
        means_before, means_after,
        output_dir / 'gmm_means_comparison.png', args.loss
    )
    log("  Saved gmm_means_after.png + comparison")

    # ── Training curve ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, args.epochs + 1), epoch_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{args.loss.upper()} Loss')
    ax.set_title(f'Decoder Fine-Tuning ({args.loss})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=150)
    plt.close(fig)

    # ── Save checkpoint ──
    ckpt_out = output_dir / 'checkpoint_finetuned.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_type': args.loss,
        'epochs': args.epochs,
        'lr': args.lr,
        'lr_end': lr_end,
        'source_checkpoint': str(args.checkpoint),
        'model_type': model_type,
    }, ckpt_out)
    log(f"  Checkpoint saved: {ckpt_out}")

    # ── Save config snapshot ──
    config_out = output_dir / 'config.yaml'
    with open(config_out, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # ── Summary ──
    total_time = time.time() - t_start
    log(f"\n{'=' * 60}")
    log("FINE-TUNING COMPLETE")
    log(f"{'=' * 60}")
    log(f"  Loss ({args.loss}): {epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f}")
    log(f"  Time: {total_time / 60:.1f} min")
    log(f"  Output: {output_dir}")
    log(f"{'=' * 60}")

    log_file.close()


if __name__ == '__main__':
    main()
