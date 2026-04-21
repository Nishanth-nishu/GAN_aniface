"""WGAN-GP specific configuration."""

from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class WGANGPConfig(BaseConfig):
    """Configuration for Wasserstein GAN with Gradient Penalty
    (Gulrajani et al., NeurIPS 2017).

    Key differences from DCGAN:
    - No BatchNorm in critic (uses LayerNorm)
    - No sigmoid in critic output
    - Gradient penalty instead of weight clipping
    - Higher critic update frequency
    """

    model_type: str = "wgan_gp"
    experiment_name: str = "wgan_gp_anime"

    # Architecture
    g_filters: int = 64
    d_filters: int = 64
    latent_dim: int = 128

    # Training (WGAN-GP paper defaults)
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    beta1: float = 0.0  # WGAN-GP uses 0.0 for beta1
    beta2: float = 0.9
    batch_size: int = 64
    num_epochs: int = 200

    # WGAN-GP specific
    gradient_penalty_weight: float = 10.0  # Lambda for GP
    critic_iterations: int = 5  # Critic updates per generator update
