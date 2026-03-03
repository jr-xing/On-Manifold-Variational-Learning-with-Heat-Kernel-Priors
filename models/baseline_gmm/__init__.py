"""
Baseline GMM clustering directly on pixel space (no neural network).
Serves as a simple baseline for comparison with VAE-based approaches.
"""

from .model import BaselineGMM, get_model

__all__ = ['BaselineGMM', 'get_model']
