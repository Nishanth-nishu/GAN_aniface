"""Configuration module for GAN architectures."""
from configs.base_config import BaseConfig
from configs.dcgan_config import DCGANConfig
from configs.wgan_gp_config import WGANGPConfig
from configs.sagan_config import SAGANConfig
from configs.use_cmhsa_config import USECMHSAConfig
from configs.stylegan2_config import StyleGAN2Config

__all__ = [
    "BaseConfig",
    "DCGANConfig",
    "WGANGPConfig",
    "SAGANConfig",
    "USECMHSAConfig",
    "StyleGAN2Config",
]
