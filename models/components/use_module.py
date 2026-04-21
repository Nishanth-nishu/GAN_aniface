"""Upsampling Squeeze-and-Excitation (USE) Module."""

import torch
import torch.nn as nn


class USEModule(nn.Module):
    """Upsampling Squeeze-and-Excitation Module.
    
    From 'Enhanced Anime Image Generation Using USE-CMHSA-GAN' (Lu, 2024).
    Combines channel attention (SE) with learned upsampling.
    """
    
    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        # Squeeze operation (Global Average Pooling)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation operation
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Upsampling operation combining nearest neighbor with conv to avoid checkerboard
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape (B, C, H, W)
        """
        batch_size, channels, _, _ = x.size()
        
        # 1. Squeeze: (B, C, 1, 1) -> (B, C)
        squeezed = self.squeeze(x).view(batch_size, channels)
        
        # 2. Excite: (B, C)
        excited = self.excitation(squeezed).view(batch_size, channels, 1, 1)
        
        # 3. Channel weighting
        weighted = x * excited.expand_as(x)
        
        # 4. Upsampling
        out = self.upsample(weighted)
        
        return out
