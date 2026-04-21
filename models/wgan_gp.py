"""Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation."""

import torch
import torch.nn as nn
from models.base_model import BaseGenerator, BaseDiscriminator
from models.dcgan import weights_init


class Generator(BaseGenerator):
    """WGAN generator using DCGAN architecture (identical to DCGAN generator)."""
    def __init__(self, latent_dim: int = 128, filters: int = 64, channels: int = 3):
        super().__init__(latent_dim)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filters * 8, filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filters, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Critic(BaseDiscriminator):
    """WGAN-GP critic.
    
    Key difference from standard discriminator:
    - NO BatchNorm (conflicts with gradient penalty)
    - Uses LayerNorm instead
    - No sigmoid activation at the end
    """
    def __init__(self, channels: int = 3, filters: int = 64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: (channels) x 64 x 64
            nn.Conv2d(channels, filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters) x 32 x 32
            nn.Conv2d(filters, filters * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([filters * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters*2) x 16 x 16
            nn.Conv2d(filters * 2, filters * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([filters * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters*4) x 8 x 8
            nn.Conv2d(filters * 4, filters * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([filters * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters*8) x 4 x 4
            nn.Conv2d(filters * 8, 1, 4, 1, 0, bias=False),
            # Linear output (no sigmoid)
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1)
