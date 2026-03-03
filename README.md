# VAE-Based Image Clustering with Geometric Priors

Official implementation for our MICCAI 2026 paper.

## Models

| Model | Description | Training Script |
|-------|-------------|-----------------|
| **VAE-GMM** | VAE with Gaussian Mixture Model prior | `train_vae_gmm.py` |
| **CLAST** | Clustering with Latent Atlas Selection & Training | `train_clast.py` |
| **Diffusion-VAE** | VAE-GMM + latent-space diffusion denoiser | `train_diffusion.py` |
| **Baseline GMM** | Pixel-space GMM clustering (non-neural) | `train_baseline.py` |
| **Baseline K-Means** | Pixel-space K-Means clustering (non-neural) | `train_baseline.py` |

## Datasets

| Dataset | Size | Classes | Type |
|---------|------|---------|------|
| MNIST | 28x28 | 10 digits | Supervised |
| Cardiac LGE | 128x128 | 5 pathologies | Supervised |
| OASIS Ventricle | 128x128 | 8 clusters | Unsupervised |

## Setup

```bash
pip install -r requirements.txt
```

MNIST auto-downloads on first run. Cardiac and OASIS data are included in `data/`.

## Training

```bash
# VAE-GMM
python train_vae_gmm.py --config configs/mnist/vae_gmm.yaml
python train_vae_gmm.py --config configs/cardiac/vae_gmm.yaml
python train_vae_gmm.py --config configs/oasis/vae_gmm.yaml

# CLAST
python train_clast.py --config configs/mnist/clast.yaml
python train_clast.py --config configs/cardiac/clast.yaml
python train_clast.py --config configs/oasis/clast.yaml

# Diffusion-VAE (two-phase: VAE training then denoiser)
python train_diffusion.py --config configs/mnist/diffusion_vae.yaml
python train_diffusion.py --config configs/cardiac/diffusion_vae.yaml

# Baselines
python train_baseline.py --config configs/mnist/baseline_gmm.yaml
python train_baseline.py --config configs/cardiac/baseline_kmeans.yaml
```

### Common CLI options

All training scripts accept:
```
--config PATH        YAML config file (required)
--output_dir PATH    Override output directory
--epochs N           Override number of epochs
--batch_size N       Override batch size
--lr FLOAT           Override learning rate
--seed INT           Override random seed
--device STR         Override device (cuda/cpu)
```

### Decoder fine-tuning (CLAST)

Fine-tune only the decoder for improved reconstruction sharpness:
```bash
python finetune_decoder.py \
    --config configs/oasis/clast.yaml \
    --checkpoint checkpoints/oasis/clast/checkpoint_best.pth \
    --output_dir results/oasis_finetuned/ \
    --loss ssim --epochs 200 --lr 5e-5
```

## Evaluation

```bash
# Supervised datasets (MNIST, Cardiac) - reports accuracy, NMI, ARI
python test.py --config configs/mnist/vae_gmm.yaml \
    --checkpoint checkpoints/mnist/vae_gmm/checkpoint_best.pth \
    --output_dir results/test_mnist_vae_gmm/

# Unsupervised dataset (OASIS) - reports silhouette, CH, DB scores
python test.py --config configs/oasis/clast.yaml \
    --checkpoint checkpoints/oasis/clast/checkpoint_best.pth \
    --output_dir results/test_oasis_clast/

# With MC uncertainty estimation
python test.py --config configs/cardiac/vae_gmm.yaml \
    --checkpoint checkpoints/cardiac/vae_gmm/checkpoint_best.pth \
    --output_dir results/test_cardiac_uncertainty/ \
    --uncertainty --n_mc 30

# Diffusion-VAE (evaluates both raw and denoised latents)
python test.py --config configs/mnist/diffusion_vae.yaml \
    --checkpoint checkpoints/mnist/diffusion_vae/checkpoint_vae.pth \
    --output_dir results/test_mnist_diffusion/
```

### Test outputs

For each run, `test.py` generates:
- `metrics.yaml` - all computed metrics
- `cluster_means.png` - decoded GMM cluster centers
- `reconstructions.png` - input vs reconstruction (neural models)
- `latent_tsne.png` - t-SNE visualization of latent space
- `cluster_purity.csv` - per-cluster label distribution (supervised)
- `uncertainty_heatmap.png` - MC uncertainty maps (if `--uncertainty`)

## Pre-trained Checkpoints

Pre-trained checkpoints are provided in `checkpoints/`:
```
checkpoints/
├── mnist/{vae_gmm,clast,diffusion_vae,baseline_gmm}/
├── cardiac/{vae_gmm,clast,diffusion_vae,baseline_gmm,baseline_kmeans}/
└── oasis/{vae_gmm,clast,diffusion_vae,baseline_gmm,baseline_kmeans}/
```

## Project Structure

```
├── train_vae_gmm.py          # VAE-GMM training
├── train_clast.py             # CLAST training
├── train_diffusion.py         # Diffusion-VAE training (two-phase)
├── train_baseline.py          # Baseline GMM/K-Means
├── finetune_decoder.py        # Post-training decoder fine-tuning
├── test.py                    # Unified evaluation & visualization
├── models/
│   ├── vae_gmm/               # VAE with GMM prior
│   ├── clast/                  # CLAST with atlas generation
│   ├── diffusion_vae/          # VAE-GMM + latent denoiser
│   ├── baseline_gmm/           # Pixel-space GMM
│   └── baseline_kmeans/        # Pixel-space K-Means
├── utils/
│   ├── config.py               # Configuration system
│   ├── data_loader.py          # Dataset loading
│   ├── metrics.py              # Clustering metrics
│   ├── visualization.py        # Plotting utilities
│   ├── atlas_visualization.py  # Atlas & cluster visualization
│   ├── logger.py               # Logging infrastructure
│   └── staged_training.py      # Multi-stage loss scheduling
├── configs/                    # YAML configs per dataset/model
├── data/                       # Dataset files
└── checkpoints/                # Pre-trained model weights
```
