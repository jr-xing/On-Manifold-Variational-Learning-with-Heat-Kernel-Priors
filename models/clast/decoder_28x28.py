"""
MNIST decoder (28×28 images) for CLAST model.
2-stage upsampling: 7×7 → 14×14 → 28×28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with dropout and batch normalization.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + identity)


class Decoder(nn.Module):
    """
    Decoder for 28×28 MNIST images.

    Architecture:
        latent_dim → FC(latent→512) → FC(512→6272) → Reshape(128, 7, 7)
        7×7 → ConvT(128→64, s=2) → ResBlock(64→64) → 14×14
        14×14 → ConvT(64→32, s=2) → ResBlock(32→32) → 28×28
        28×28 → Conv(32→1) → Sigmoid → [B, 1, 28, 28]
    """
    def __init__(self, latent_dim):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128 * 7 * 7)  # 512 → 6272

        # Stage 1: 7×7 → 14×14
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.res1 = ResidualBlock(64, 64)

        # Stage 2: 14×14 → 28×28
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.res2 = ResidualBlock(32, 32)

        # Output layer
        self.conv_out = nn.Conv2d(32, 1, 3, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, z):
        """
        Args:
            z: Latent code [B, latent_dim]

        Returns:
            recon: Reconstructed image [B, 1, 28, 28]
        """
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)

        x = self.up1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        return self.act(self.conv_out(x))
