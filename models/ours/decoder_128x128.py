"""
Cardiac decoder (128×128 images) for CLAST model.
5-stage upsampling: 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
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
    Decoder for 128×128 cardiac images.

    Architecture:
        latent_dim → FC(latent→512) → FC(512→8192) → Reshape(512, 4, 4)
        4×4 → ConvT(512→512, s=2) → ResBlock(512→512) → 8×8
        8×8 → ConvT(512→256, s=2) → ResBlock(256→256) → 16×16
        16×16 → ConvT(256→128, s=2) → ResBlock(128→128) → 32×32
        32×32 → ConvT(128→64, s=2) → ResBlock(64→64) → 64×64
        64×64 → ConvT(64→32, s=2) → ResBlock(32→32) → 128×128
        128×128 → Conv(32→1) → Sigmoid → [B, 1, 128, 128]
    """
    def __init__(self, latent_dim):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512 * 4 * 4)  # 512 → 8192

        # Stage 1: 4×4 → 8×8
        self.up1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.res1 = ResidualBlock(512, 512)

        # Stage 2: 8×8 → 16×16
        self.up2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.res2 = ResidualBlock(256, 256)

        # Stage 3: 16×16 → 32×32
        self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.res3 = ResidualBlock(128, 128)

        # Stage 4: 32×32 → 64×64
        self.up4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.res4 = ResidualBlock(64, 64)

        # Stage 5: 64×64 → 128×128
        self.up5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.res5 = ResidualBlock(32, 32)

        # Output layer
        self.conv_out = nn.Conv2d(32, 1, 3, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, z):
        """
        Args:
            z: Latent code [B, latent_dim]

        Returns:
            recon: Reconstructed image [B, 1, 128, 128]
        """
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 512, 4, 4)

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
