"""
Decoder for 28×28 images (e.g., MNIST).
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


class Decoder(nn.Module):
    """Decoder network for MNIST (latent_dim → 28×28)."""

    def __init__(self, latent_dim=10):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128 * 7 * 7)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7×7 → 14×14
        self.res1 = ResidualBlock(64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 14×14 → 28×28
        self.res2 = ResidualBlock(32, 32)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)  # Reshape back to feature map
        x = F.relu(self.up1(x))
        x = self.res1(x)
        x = F.relu(self.up2(x))
        x = self.res2(x)
        return self.act(self.conv_out(x))
