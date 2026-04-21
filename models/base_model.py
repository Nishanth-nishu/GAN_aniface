"""Base classes for Generator and Discriminator."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseGenerator(nn.Module, ABC):
    """Abstract base class for all GAN generators."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Latent noise vector of shape (batch_size, latent_dim, 1, 1) or 
               (batch_size, latent_dim) depending on architecture.
               
        Returns:
            Generated image tensor of shape (batch, channels, height, width).
        """
        pass
        
    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample latent vectors from the prior (usually standard normal).
        
        Args:
            batch_size: Number of vectors to sample.
            device: Computation device.
            
        Returns:
            Tensor of shape (batch_size, latent_dim, 1, 1).
        """
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=device)
        
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Convenience method to sample latent vectors and generate images."""
        z = self.sample_latent(batch_size, device)
        return self(z)


class BaseDiscriminator(nn.Module, ABC):
    """Abstract base class for all GAN discriminators (or critics)."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Real or generated image tensor of shape (batch, channels, h, w).
            
        Returns:
            Score tensor (shape varies by architecture, usually (batch, 1) or 
            (batch, 1, 1, 1)).
        """
        pass
