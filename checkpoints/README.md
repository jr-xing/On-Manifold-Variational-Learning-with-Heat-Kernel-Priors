# Pre-trained Checkpoints

This directory contains pre-trained model weights for all experiments.

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
    ├── diffusion_vae/
    ├── baseline_gmm/
    └── baseline_kmeans/
```

Each subdirectory contains a `checkpoint_best.pth` file used by `test.py` and the evaluation notebooks in `notebooks/`.
