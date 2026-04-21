"""Self-Attention GAN (SAGAN) implementation (Zhang et al., 2019)."""

import torch
import torch.nn as nn
from models.base_model import BaseGenerator, BaseDiscriminator
from models.components.self_attention import SelfAttention
from models.components.spectral_norm import SpectralNorm
from models.dcgan import weights_init


class Generator(BaseGenerator):
    """SAGAN Generator.
    
    Like DCGAN, but places a Self-Attention module after a specified layer.
    """
    def __init__(
        self, 
        latent_dim: int = 128, 
        filters: int = 64, 
        channels: int = 3,
        attention_layer: int = 2
    ):
        super().__init__(latent_dim)
        
        # Layer 0
        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True)
        )
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(filters * 8, filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True)
        )
        
        # Layer 4 (Output)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(filters, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.attention_layer = attention_layer
        
        # Define attention depending on placement
        if attention_layer == 1:
            self.attn = SelfAttention(filters * 4)
        elif attention_layer == 2:
            self.attn = SelfAttention(filters * 2)
        elif attention_layer == 3:
            self.attn = SelfAttention(filters)
        else:
            self.attn = None

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.layer0(z)
        
        out = self.layer1(out)
        if self.attention_layer == 1 and self.attn is not None:
            out = self.attn(out)
            
        out = self.layer2(out)
        if self.attention_layer == 2 and self.attn is not None:
            out = self.attn(out)
            
        out = self.layer3(out)
        if self.attention_layer == 3 and self.attn is not None:
            out = self.attn(out)
            
        out = self.layer4(out)
        return out


class Discriminator(BaseDiscriminator):
    """SAGAN Discriminator (Critic).
    
    Uses Spectral Normalization on ALL convolutional layers.
    Places a Self-Attention module after a specified layer.
    """
    def __init__(
        self, 
        channels: int = 3, 
        filters: int = 64,
        attention_layer: int = 1
    ):
        super().__init__()
        
        # Layer 0
        self.layer0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(channels, filters, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 1
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(filters, filters * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(filters * 2, filters * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(filters * 4, filters * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 4 (Output)
        # Note: No sigmoid because SAGAN uses Hinge loss
        self.layer4 = SpectralNorm(nn.Conv2d(filters * 8, 1, 4, 1, 0, bias=False))
        
        self.attention_layer = attention_layer
        
        # Define attention depending on placement
        if attention_layer == 1:
            self.attn = SelfAttention(filters * 2)
        elif attention_layer == 2:
            self.attn = SelfAttention(filters * 4)
        elif attention_layer == 3:
            self.attn = SelfAttention(filters * 8)
        else:
            self.attn = None

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer0(x)
        
        out = self.layer1(out)
        if self.attention_layer == 1 and self.attn is not None:
            out = self.attn(out)
            
        out = self.layer2(out)
        if self.attention_layer == 2 and self.attn is not None:
            out = self.attn(out)
            
        out = self.layer3(out)
        if self.attention_layer == 3 and self.attn is not None:
            out = self.attn(out)
            
        out = self.layer4(out)
        return out.view(-1, 1)
