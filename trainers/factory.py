"""Trainer Factory."""

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainers.gan_trainer import GANTrainer
from trainers.wgan_trainer import WGANGPTrainer


def create_trainer(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    config: Any,
    device: torch.device
) -> Any:
    """Create the appropriate trainer based on configuration.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        dataloader: Training dataloader.
        config: Subclass of BaseConfig.
        device: Computation device.
        
    Returns:
        Configured trainer instance.
    """
    # Setup Optimizers
    opt_g = torch.optim.Adam(
        generator.parameters(), 
        lr=config.learning_rate_g, 
        betas=(config.beta1, config.beta2)
    )
    
    opt_d = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config.learning_rate_d, 
        betas=(config.beta1, config.beta2)
    )
    
    model_type = config.model_type.lower()
    
    if model_type in ["dcgan", "sagan", "use_cmhsa"]:
        return GANTrainer(
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            dataloader=dataloader,
            config=config,
            device=device
        )
    elif model_type == "wgan_gp":
        return WGANGPTrainer(
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            dataloader=dataloader,
            config=config,
            device=device
        )
    else:
        raise ValueError(f"No local trainer available for model_type: {model_type}")
