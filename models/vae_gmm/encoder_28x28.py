"""
Encoder for 28×28 images (e.g., MNIST).
Part of the VAE-GMM model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block without skip connection."""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)  # No skip connection


class Encoder(nn.Module):
    """Encoder network for MNIST (28×28 → latent_dim)."""

    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()
        self.res1 = ResidualBlock(1, 32)
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 28×28 → 14×14
        self.res2 = ResidualBlock(64, 64)
        self.downsample2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14×14 → 7×7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.res1(x)
        x = F.relu(self.downsample1(x))
        x = self.res2(x)
        x = F.relu(self.downsample2(x))
        x_flattened = self.flatten(x)
        x_fc = F.relu(self.fc1(x_flattened))
        return self.fc_mu(x_fc), self.fc_logvar(x_fc)
