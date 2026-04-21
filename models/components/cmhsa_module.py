"""Convolution-Multi-Head Self-Attention (CMHSA) Module."""

import torch
import torch.nn as nn


class CMHSAModule(nn.Module):
    """Convolution-based Multi-Head Self-Attention.
    
    From 'Enhanced Anime Image Generation Using USE-CMHSA-GAN' (Lu, 2024).
    Uses 1x1 convolutions for Q, K, V projections instead of linear layers
    to maintain spatial relationships before computing attention.
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable skip connection weight
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape (B, C, H, W)
        """
        B, C, H, W = x.size()
        N = H * W
        
        # Compute Q, K, V
        # Shape: (B, C, H, W) -> (B, num_heads, head_dim, N)
        q = self.q_conv(x).view(B, self.num_heads, self.head_dim, N)
        k = self.k_conv(x).view(B, self.num_heads, self.head_dim, N)
        v = self.v_conv(x).view(B, self.num_heads, self.head_dim, N)
        
        # Compute attention scores
        # q.transpose(-2, -1): (B, num_heads, N, head_dim)
        # k: (B, num_heads, head_dim, N)
        # attn: (B, num_heads, N, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        # attn: (B, num_heads, N, N)
        # v.transpose(-2, -1): (B, num_heads, N, head_dim)
        # out: (B, num_heads, N, head_dim)
        out = (attn @ v.transpose(-2, -1))
        
        # Reshape to (B, C, H, W)
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        # Final projection and residual connection
        out = self.proj(out)
        
        return x + self.gamma * out
