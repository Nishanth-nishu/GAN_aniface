"""Self-Attention module for SAGAN."""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Self-Attention layer for spatial dimension.
    
    As proposed in SAGAN (Self-Attention Generative Adversarial Networks)
    by Zhang et al. (2019).
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # 1x1 convolutions for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable attention weight parameter (starts at 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape (batch, channels, height, width)
            
        Returns:
            Attention map applied to input.
        """
        batch_size, channels, width, height = x.size()
        
        # Query: (B, C/8, H*W)
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        
        # Key: (B, C/8, H*W)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        
        # Energy: (B, H*W, H*W)
        energy = torch.bmm(proj_query, proj_key)
        
        # Attention map: (B, H*W, H*W)
        attention = self.softmax(energy)
        
        # Value: (B, C, H*W)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Output: (B, C, H*W) -> (B, C, H, W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        
        # Combine with original input using learnable weight
        out = self.gamma * out + x
        
        return out
