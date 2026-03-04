"""
Cardiac encoder (128×128 images) for CLAST model.
5-stage downsampling: 128×128 → 64×64 → 32×32 → 16×16 → 8×8 → 4×4
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


class Encoder(nn.Module):
    """
    Encoder for 128×128 cardiac images.

    Architecture:
        128×128 → ResBlock(1→32) → Conv(32→64, s=2) → 64×64
        64×64 → ResBlock(64→64) → Conv(64→128, s=2) → 32×32
        32×32 → ResBlock(128→128) → Conv(128→256, s=2) → 16×16
        16×16 → ResBlock(256→256) → Conv(256→512, s=2) → 8×8
        8×8 → ResBlock(512→512) → Conv(512→512, s=2) → 4×4
        4×4 → Flatten → FC(8192→512) → FC(512→latent_dim)
    """
    def __init__(self, latent_dim):
        super().__init__()
        # Stage 1: 128×128 → 64×64
        self.res1 = ResidualBlock(1, 32)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)

        # Stage 2: 64×64 → 32×32
        self.res2 = ResidualBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)

        # Stage 3: 32×32 → 16×16
        self.res3 = ResidualBlock(128, 128)
        self.down3 = nn.Conv2d(128, 256, 4, 2, 1)

        # Stage 4: 16×16 → 8×8
        self.res4 = ResidualBlock(256, 256)
        self.down4 = nn.Conv2d(256, 512, 4, 2, 1)

        # Stage 5: 8×8 → 4×4
        self.res5 = ResidualBlock(512, 512)
        self.down5 = nn.Conv2d(512, 512, 4, 2, 1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 4, 512)  # 8192 → 512
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 1, 128, 128]

        Returns:
            mu: Mean [B, latent_dim]
            logvar: Log-variance [B, latent_dim]
        """
        x = self.res1(x)
        x = F.relu(self.down1(x))

        x = self.res2(x)
        x = F.relu(self.down2(x))

        x = self.res3(x)
        x = F.relu(self.down3(x))

        x = self.res4(x)
        x = F.relu(self.down4(x))

        x = self.res5(x)
        x = F.relu(self.down5(x))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))

        return self.fc_mu(x), self.fc_logvar(x)
