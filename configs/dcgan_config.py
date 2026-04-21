"""DCGAN-specific configuration."""

from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class DCGANConfig(BaseConfig):
    """Configuration for Deep Convolutional GAN (Radford et al., 2016).

    Uses standard BCE adversarial loss with BatchNorm in both
    generator and discriminator.
    """

    model_type: str = "dcgan"
    experiment_name: str = "dcgan_anime"

    # Architecture
    g_filters: int = 64
    d_filters: int = 64
    latent_dim: int = 128

    # Training (DCGAN paper defaults)
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 128
    num_epochs: int = 200

    # Loss
    label_smoothing: float = 0.1  # One-sided label smoothing for stability
