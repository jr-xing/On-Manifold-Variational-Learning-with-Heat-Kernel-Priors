"""
Baseline K-Means clustering on raw pixel space.

This is a non-neural baseline that:
- Operates directly on pixel space (784D for MNIST, 16384D for 128x128)
- Uses sklearn K-Means without any gradient-based training
- Serves as a simple baseline for comparison with VAE approaches
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATASET_TO_INPUT_DIM = {
    'mnist': 784,       # 28 * 28
    'cardiac': 16384,   # 128 * 128
    'oasis': 16384,     # 128 * 128
}


class BaselineKMeans(nn.Module):
    """
    Baseline K-Means model for clustering in pixel space.

    Note: This is not a real neural network, but inherits from nn.Module
    for compatibility with the training framework.
    """

    def __init__(self, input_dim=784, num_clusters=10):
        super(BaselineKMeans, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.scaler = StandardScaler()
        self.kmeans = None

        self.image_size = int(np.sqrt(input_dim))
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Dummy forward pass for compatibility. Returns (recon, mu, logvar, z)."""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        recon = x_flat.view(batch_size, 1, self.image_size, self.image_size)
        return recon, None, None, x_flat

    def compute_loss(self, x, recon, mu, logvar, z, loss_weights):
        """Dummy loss (no gradient descent for baseline)."""
        return {
            'total_loss': torch.tensor(0.0, device=x.device, requires_grad=True),
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'gmm_loss': 0.0
        }

    def extract_latent_features(self, data_loader, device):
        """Extract flattened pixel features (no latent space)."""
        self.eval()
        features_list, labels_list = [], []

        with torch.no_grad():
            for batch, target in data_loader:
                batch_flat = batch.view(batch.size(0), -1).cpu().numpy()
                features_list.append(batch_flat)
                labels_list.append(target.numpy())

        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        features_normalized = self.scaler.fit_transform(features)

        # Sort lexicographically for deterministic K-Means initialization
        sort_idx = np.lexsort(features_normalized.T[::-1])
        return features_normalized[sort_idx], labels[sort_idx]

    def fit_gmm_and_evaluate(self, z_data, labels, covariance_type='diag', n_init=10):
        """Fit K-Means on normalized pixel space."""
        self.kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=0,
            max_iter=300,
            n_init=10
        )
        kmeans_labels = self.kmeans.fit_predict(z_data)
        return {'gmm': self.kmeans, 'gmm_labels': kmeans_labels}


def get_model(config):
    """Factory function to create BaselineKMeans model instance."""
    dataset_name = config['dataset']['name']
    input_dim = DATASET_TO_INPUT_DIM.get(dataset_name)
    if input_dim is None:
        raise ValueError(f"Baseline K-Means not configured for dataset: {dataset_name}")

    return BaselineKMeans(
        input_dim=input_dim,
        num_clusters=config['model']['num_clusters']
    )
