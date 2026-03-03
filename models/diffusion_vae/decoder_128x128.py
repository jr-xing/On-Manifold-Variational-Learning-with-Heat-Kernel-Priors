"""
Decoder for 128×128 images (e.g., Cardiac).
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
    """Decoder network for Cardiac (latent_dim → 128×128)."""

    def __init__(self, latent_dim=10):
        super(Decoder, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512 * 4 * 4)  # 512 → 8192

        # Stage 1: 4×4 → 8×8
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(512, 512)

        # Stage 2: 8×8 → 16×16
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualBlock(256, 256)

        # Stage 3: 16×16 → 32×32
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.res3 = ResidualBlock(128, 128)

        # Stage 4: 32×32 → 64×64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.res4 = ResidualBlock(64, 64)

        # Stage 5: 64×64 → 128×128
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.res5 = ResidualBlock(32, 32)

        # Output layer
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, z):
        # Project to feature map
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 512, 4, 4)  # Reshape to 512×4×4

        # Upsample: 4 → 8 → 16 → 32 → 64 → 128
        x = F.relu(self.up1(x))
        x = self.res1(x)

        x = F.relu(self.up2(x))
        x = self.res2(x)

        x = F.relu(self.up3(x))
        x = self.res3(x)

        x = F.relu(self.up4(x))
        x = self.res4(x)

        x = F.relu(self.up5(x))
        x = self.res5(x)

        return self.act(self.conv_out(x))
