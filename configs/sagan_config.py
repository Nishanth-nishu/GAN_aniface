"""SAGAN-specific configuration."""

from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class SAGANConfig(BaseConfig):
    """Configuration for Self-Attention GAN (Zhang et al., ICML 2019).

    Key innovations:
    - Self-attention mechanism for long-range dependencies
    - Spectral normalization on discriminator weights
    - Two-timescale update rule (TTUR)
    - Hinge loss instead of BCE
    """

    model_type: str = "sagan"
    experiment_name: str = "sagan_anime"

    # Architecture
    g_filters: int = 64
    d_filters: int = 64
    latent_dim: int = 128
    num_attention_heads: int = 1

    # TTUR: different learning rates for G and D
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 4e-4
    beta1: float = 0.0
    beta2: float = 0.9
    batch_size: int = 64
    num_epochs: int = 200

    # SAGAN specific
    spectral_norm: bool = True
    attention_layer_g: int = 2  # Insert attention after this generator layer
    attention_layer_d: int = 1  # Insert attention after this discriminator layer
