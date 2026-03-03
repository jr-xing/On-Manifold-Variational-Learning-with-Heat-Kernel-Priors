"""
Encoder for 128×128 images (e.g., Cardiac).
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
    """Encoder network for Cardiac (128×128 → latent_dim)."""

    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()
        # Stage 1: 128×128 → 64×64
        self.res1 = ResidualBlock(1, 32)
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        # Stage 2: 64×64 → 32×32
        self.res2 = ResidualBlock(64, 64)
        self.downsample2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        # Stage 3: 32×32 → 16×16
        self.res3 = ResidualBlock(128, 128)
        self.downsample3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Stage 4: 16×16 → 8×8
        self.res4 = ResidualBlock(256, 256)
        self.downsample4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # Stage 5: 8×8 → 4×4
        self.res5 = ResidualBlock(512, 512)
        self.downsample5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 4, 512)  # 8192 → 512
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        # Downsample: 128 → 64 → 32 → 16 → 8 → 4
        x = self.res1(x)
        x = F.relu(self.downsample1(x))

        x = self.res2(x)
        x = F.relu(self.downsample2(x))

        x = self.res3(x)
        x = F.relu(self.downsample3(x))

        x = self.res4(x)
        x = F.relu(self.downsample4(x))

        x = self.res5(x)
        x = F.relu(self.downsample5(x))

        # Flatten and project to latent space
        x_flattened = self.flatten(x)
        x_fc = F.relu(self.fc1(x_flattened))
        mu = self.fc_mu(x_fc)
        logvar = self.fc_logvar(x_fc)

        # Clamp logvar to prevent numerical overflow (exp(0.5 * logvar) can cause NaN)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar
