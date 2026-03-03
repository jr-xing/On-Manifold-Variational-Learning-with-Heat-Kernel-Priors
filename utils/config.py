"""
Configuration management for VAE-GMM training framework.
Handles YAML config parsing, CLI argument parsing, and default values.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'experiment': {
            'name': 'experiment',
            'seed': 42,
            'results_dir': './results'
        },
        'model': {
            'name': 'vae_gmm_28x28',
            'latent_dim': 10,
            'num_clusters': None  # Auto-inferred from dataset
        },
        'dataset': {
            'name': 'mnist',
            'root': './data',
            'batch_size': 1024,
            'num_workers': 0,
            'shuffle': True
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'device': 'cuda',
            'eval_gmm_covariance': 'diag'
        },
        'logging': {
            'use_tqdm': True,
            'log_interval': 10,
            'save_metrics': True,
            'visualization_interval': 10
        },
        'loss_weights': {
            'reconstruction': 1.0,
            'kl': 0.1,
            'gmm': 1.0,
            'trajectory': 0.02,
            'jacobian': 0.01
        },
        'staged_training': {
            'enabled': False,
            'stage1_end_epoch': 50,
            'stage2_end_epoch': 100,
            'stage1_weights': {
                'reconstruction': 1.0,
                'kl': 0.01,
                'gmm': 0.0
            },
            'stage2_weights': {
                'reconstruction': 1.0,
                'kl': 0.1,
                'gmm': 0.5
            },
            'adaptive_monitoring': {
                'enabled': True,
                'recon_critical_threshold': 5470,
                'recon_degradation_tolerance': 500,
                'rebalance_factor': 1.5,
                'check_interval': 5
            }
        }
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VAE-GMM models for clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # Output directory
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    # Other arguments
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g., cuda:0, cuda:1, cpu)')

    return parser.parse_args()


def merge_configs(base_config: Dict[str, Any],
                  cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI arguments into config (CLI takes precedence)."""
    config = base_config.copy()

    # Override dataset settings
    if cli_args.batch_size is not None:
        config['dataset']['batch_size'] = cli_args.batch_size

    # Override training settings
    if cli_args.epochs is not None:
        config['training']['epochs'] = cli_args.epochs
    if cli_args.lr is not None:
        config['training']['learning_rate'] = cli_args.lr

    # Override experiment settings
    if cli_args.output_dir is not None:
        config['experiment']['results_dir'] = cli_args.output_dir
    if cli_args.seed is not None:
        config['experiment']['seed'] = cli_args.seed

    # Override device
    if cli_args.device is not None:
        config['training']['device'] = cli_args.device

    return config


def infer_num_clusters(dataset_name: str) -> int:
    """
    Infer number of clusters based on dataset.

    Args:
        dataset_name: Name of the dataset ('mnist', 'cardiac', or 'oasis').

    Returns:
        Number of clusters.
    """
    if dataset_name == 'mnist':
        return 10
    elif dataset_name == 'cardiac':
        return 5
    elif dataset_name == 'oasis':
        return 10  # Default for unsupervised clustering
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of: 'mnist', 'cardiac', 'oasis'")


def get_config() -> Dict[str, Any]:
    """
    Main entry point for configuration loading.
    Priority: CLI args > Config file > Defaults
    """
    # Parse CLI arguments
    args = parse_args()

    # Start with defaults
    if args.config is not None:
        # Load from config file
        config = load_yaml_config(args.config)
        # Fill in missing values from defaults
        default_config = get_default_config()
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            elif isinstance(default_config[key], dict):
                for subkey in default_config[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = default_config[key][subkey]
    else:
        config = get_default_config()

    # Override with CLI arguments
    config = merge_configs(config, args)

    # Infer num_clusters if not set
    if config['model']['num_clusters'] is None:
        config['model']['num_clusters'] = infer_num_clusters(config['dataset']['name'])

    # Auto-detect device
    import torch
    if config['training']['device'] == 'cuda' and not torch.cuda.is_available():
        config['training']['device'] = 'cpu'
        print("CUDA not available, using CPU")

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
