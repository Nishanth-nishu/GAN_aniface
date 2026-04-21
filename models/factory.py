"""Factory for instantiating GAN models."""

from typing import Tuple, Dict, Any, Type
import torch.nn as nn

from models.dcgan import Generator as DCGAN_G, Discriminator as DCGAN_D
from models.wgan_gp import Generator as WGAN_G, Critic as WGAN_C
from models.sagan import Generator as SAGAN_G, Discriminator as SAGAN_D
from models.use_cmhsa_gan import Generator as USE_G, Discriminator as USE_D


def create_model(config) -> Tuple[nn.Module, nn.Module]:
    """Create Generator and Discriminator based on configuration.
    
    Args:
        config: Subclass of BaseConfig (e.g. DCGANConfig).
        
    Returns:
        Tuple of (Generator, Discriminator).
    """
    model_type = config.model_type.lower()
    
    if model_type == "dcgan":
        gen = DCGAN_G(
            latent_dim=config.latent_dim,
            filters=config.g_filters,
            channels=config.channels
        )
        disc = DCGAN_D(
            channels=config.channels,
            filters=config.d_filters
        )
        
    elif model_type == "wgan_gp":
        gen = WGAN_G(
            latent_dim=config.latent_dim,
            filters=config.g_filters,
            channels=config.channels
        )
        disc = WGAN_C(
            channels=config.channels,
            filters=config.d_filters
        )
        
    elif model_type == "sagan":
        gen = SAGAN_G(
            latent_dim=config.latent_dim,
            filters=config.g_filters,
            channels=config.channels,
            attention_layer=config.attention_layer_g
        )
        disc = SAGAN_D(
            channels=config.channels,
            filters=config.d_filters,
            attention_layer=config.attention_layer_d
        )
        
    elif model_type == "use_cmhsa":
        gen = USE_G(
            latent_dim=config.latent_dim,
            filters=config.g_filters,
            channels=config.channels,
            se_reduction_ratio=config.se_reduction_ratio,
            num_heads=config.num_attention_heads
        )
        disc = USE_D(
            channels=config.channels,
            filters=config.d_filters
        )
        
    elif model_type == "stylegan2":
        # StyleGAN2 requires its own complex repo and wrapper.
        # We'll implement a standalone wrapper for it later,
        # but for the core local architectures, we use the above.
        raise NotImplementedError("StyleGAN2 requires custom runner.")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return gen, disc
