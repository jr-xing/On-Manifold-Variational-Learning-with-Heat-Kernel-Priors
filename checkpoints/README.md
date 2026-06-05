# Pre-trained Checkpoints

This directory contains the bundled pre-trained model weights used by the
evaluation scripts and notebooks.

## Download

Download `checkpoints.zip` from the [OSF repository](https://osf.io/879yr/files/osfstorage?view_only=ee816762b1b74632a65e9eab1fe7a706) and unzip it into the repository root:

```bash
unzip checkpoints.zip -d .
```

## Expected Structure

```
checkpoints/
├── mnist/
│   ├── vae_gmm/
│   ├── ours/
│   ├── diffusion_vae/
│   └── baseline_gmm/
├── cardiac/
│   ├── vae_gmm/
│   ├── ours/
│   ├── diffusion_vae/
│   ├── baseline_gmm/
│   └── baseline_kmeans/
└── oasis/
    ├── vae_gmm/
    ├── ours/
    └── diffusion_vae/
```

Neural models use `checkpoint_best.pth`, except Diffusion-VAE, which stores the
frozen VAE-GMM as `checkpoint_vae.pth` and its latent denoiser as `denoiser.pth`.
OASIS Ours also includes `checkpoint_finetuned.pth` for the decoder-finetuned
model used in reconstruction/sharpness analyses.

Classical baseline models are tiny and can be refit from the YAML configs with
`train_baseline.py`. The bundle includes the reported MNIST GMM and cardiac
GMM/K-Means checkpoints; MNIST K-Means and OASIS classical baselines are not
distributed as pre-trained weights.

See `MANIFEST.tsv` for file sizes and SHA256 checksums.
