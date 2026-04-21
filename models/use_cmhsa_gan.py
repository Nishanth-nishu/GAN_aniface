"""USE-CMHSA-GAN implementation (Lu, 2024).

Implementation of "Enhanced Anime Image Generation Using USE-CMHSA-GAN".
"""

import torch
import torch.nn as nn
from models.base_model import BaseGenerator
from models.dcgan import Discriminator as DCGANDiscriminator
from models.dcgan import weights_init
from models.components.use_module import USEModule
from models.components.cmhsa_module import CMHSAModule


class Generator(BaseGenerator):
    """USE-CMHSA Generator.
    
    Replaces intermediate DeConv layers of DCGAN with USE Modules,
    and inserts CMHSA Modules for multi-head self-attention.
    """
    def __init__(
        self, 
        latent_dim: int = 128, 
        filters: int = 64, 
        channels: int = 3,
        se_reduction_ratio: int = 16,
        num_heads: int = 4
    ):
        super().__init__(latent_dim)
        
        # Initial projection: (latent_dim) x 1 x 1 -> (filters*8) x 4 x 4
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True)
        )
        
        # Layer 1: USE Module instead of DeConv
        # (filters*8) -> (filters*4), 4x4 -> 8x8
        self.use1 = USEModule(filters * 8, filters * 4, se_reduction_ratio)
        
        # CMHSA Module 1
        self.cmhsa1 = CMHSAModule(filters * 4, num_heads)
        
        # Layer 2: USE Module
        # (filters*4) -> (filters*2), 8x8 -> 16x16
        self.use2 = USEModule(filters * 4, filters * 2, se_reduction_ratio)
        
        # CMHSA Module 2
        self.cmhsa2 = CMHSAModule(filters * 2, num_heads)
        
        # Layer 3: USE Module
        # (filters*2) -> (filters), 16x16 -> 32x32
        self.use3 = USEModule(filters * 2, filters, se_reduction_ratio)
        
        # Final Layer: Standard DeConv to 3 channels
        # (filters) -> (channels), 32x32 -> 64x64
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.initial(z)
        
        out = self.use1(out)
        out = self.cmhsa1(out)
        
        out = self.use2(out)
        out = self.cmhsa2(out)
        
        out = self.use3(out)
        
        out = self.final(out)
        return out


# The USE-CMHSA-GAN paper uses a standard DCGAN discriminator
Discriminator = DCGANDiscriminator
