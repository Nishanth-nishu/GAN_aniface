"""USE-CMHSA-GAN specific configuration."""

from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class USECMHSAConfig(BaseConfig):
    """Configuration for USE-CMHSA-GAN (Lu, arXiv 2024).

    Architecture innovations over DCGAN:
    - USE Module: Upsampling Squeeze-and-Excitation for channel attention
    - CMHSA Module: Convolution-based Multi-Head Self-Attention for
      spatial coherence
    - Replaces DeConv layers with USE modules in generator
    - CMHSA inserted between generator layers
    - Discriminator unchanged from DCGAN
    """

    model_type: str = "use_cmhsa"
    experiment_name: str = "use_cmhsa_anime"

    # Architecture
    g_filters: int = 64
    d_filters: int = 64
    latent_dim: int = 128

    # Training
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 64
    num_epochs: int = 200

    # USE module
    se_reduction_ratio: int = 16  # Squeeze-and-Excitation reduction ratio

    # CMHSA module
    num_attention_heads: int = 4  # Multi-head self-attention heads
    cmhsa_layers: int = 2  # Number of CMHSA modules in generator
