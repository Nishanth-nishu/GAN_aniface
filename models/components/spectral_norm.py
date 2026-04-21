"""Spectral Normalization module.

Implementation of Spectral Normalization for GANs (Miyato et al., 2018).
PyTorch's built-in spectral_norm is usually preferred, but this provides
a customizable wrapper for clarity.
"""

import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def SpectralNorm(module: nn.Module) -> nn.Module:
    """Wrapper for PyTorch's spectral_norm.
    
    Args:
        module: A PyTorch neural network layer (e.g., nn.Conv2d)
        
    Returns:
        The module with spectral normalization applied.
    """
    return spectral_norm(module)
