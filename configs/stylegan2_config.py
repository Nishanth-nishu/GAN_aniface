"""StyleGAN2-ADA specific configuration."""

from dataclasses import dataclass
from typing import Optional
from configs.base_config import BaseConfig


@dataclass
class StyleGAN2Config(BaseConfig):
    """Configuration for StyleGAN2-ADA (Karras et al., NeurIPS 2020).

    State-of-the-art GAN architecture with:
    - Style-based generator with mapping network
    - Adaptive discriminator augmentation (ADA)
    - Path length regularization
    - R1 regularization
    - Exponential moving average of generator weights
    """

    model_type: str = "stylegan2"
    experiment_name: str = "stylegan2_anime"

    # Architecture — StyleGAN2 uses higher resolution
    image_size: int = 256
    latent_dim: int = 512
    mapping_layers: int = 8  # Mapping network depth
    style_dim: int = 512

    # Training
    learning_rate_g: float = 2.5e-3
    learning_rate_d: float = 2.5e-3
    beta1: float = 0.0
    beta2: float = 0.99
    batch_size: int = 16  # Smaller batch for higher resolution
    num_epochs: int = 100

    # StyleGAN2 specific
    r1_gamma: float = 10.0  # R1 regularization weight
    path_length_regularization: float = 2.0
    ema_decay: float = 0.999  # EMA for generator
    ada_target: float = 0.6  # Target augmentation probability
    ada_length: int = 500000  # Kimg to reach target

    # Transfer learning
    pretrained_url: Optional[str] = None  # URL to pre-trained .pkl
    freeze_d_layers: int = 0  # Number of discriminator layers to freeze
