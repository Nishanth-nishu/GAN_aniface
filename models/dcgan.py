"""Deep Convolutional GAN (DCGAN) implementation (Radford et al., 2016)."""

import torch
import torch.nn as nn
from models.base_model import BaseGenerator, BaseDiscriminator


def weights_init(m):
    """Custom weights initialization called on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(BaseGenerator):
    def __init__(self, latent_dim: int = 128, filters: int = 64, channels: int = 3):
        super().__init__(latent_dim)
        
        self.main = nn.Sequential(
            # Input: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim, filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True),
            
            # State: (filters*8) x 4 x 4
            nn.ConvTranspose2d(filters * 8, filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),
            
            # State: (filters*4) x 8 x 8
            nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),
            
            # State: (filters*2) x 16 x 16
            nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            
            # State: (filters) x 32 x 32
            nn.ConvTranspose2d(filters, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (channels) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(BaseDiscriminator):
    def __init__(self, channels: int = 3, filters: int = 64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: (channels) x 64 x 64
            nn.Conv2d(channels, filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters) x 32 x 32
            nn.Conv2d(filters, filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters*2) x 16 x 16
            nn.Conv2d(filters * 2, filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters*4) x 8 x 8
            nn.Conv2d(filters * 4, filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (filters*8) x 4 x 4
            nn.Conv2d(filters * 8, 1, 4, 1, 0, bias=False),
            # Note: No sigmoid here! We use BCEWithLogitsLoss for numerical stability
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape is (batch_size, 1, 1, 1), reshape to (batch_size, 1)
        return self.main(x).view(-1, 1)
