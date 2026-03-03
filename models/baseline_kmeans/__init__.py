"""
Baseline K-Means clustering directly on pixel space (no neural network).
Serves as a simple baseline for comparison with VAE-based approaches.
"""

from .model import BaselineKMeans, get_model

__all__ = ['BaselineKMeans', 'get_model']
