"""FID computation using clean-fid library."""

import os
from glob import glob
from pathlib import Path
import logging
from cleanfid import fid
import torch

from configs.base_config import BaseConfig

logger = logging.getLogger(__name__)


def compute_fid(
    config: BaseConfig,
    generator: torch.nn.Module,
    num_samples: int = 10000,
    dataset_name: str = "anime_faces",
    device: torch.device = torch.device("cuda")
) -> float:
    """Computes FID using clean-fid.
    
    clean-fid offers more robust resizing and avoids aliasing artifacts
    compared to the original tensorflow or pytorch-fid implementation.
    
    Args:
        config: GAN Configuration.
        generator: Trained generator model.
        num_samples: Number of samples to use.
        dataset_name: Name for the cached dataset statistics.
        device: Computation device.
        
    Returns:
        FID score.
    """
    logger.info("Starting FID computation...")
    
    # 1. First, make sure real dataset statistics are computed and cached
    real_images_dir = config.data_dir
    # Check if we have pre-computed stats using clean-fid custom dataset logic
    # Clean-fid handles caching automatically internally when we use custom datasets
    
    if not os.path.exists(real_images_dir) or len(glob(os.path.join(real_images_dir, "*.*"))) == 0:
         logger.warning(f"Real data directory {real_images_dir} not found or empty. Cannot compute FID.")
         return float("inf")
         
    # Register dataset if not already registered (this computes/caches stats if missing)
    try:
        fid.make_custom_stats(
            dataset_name, 
            real_images_dir, 
            mode="clean",
            num=min(num_samples, len(glob(os.path.join(real_images_dir, "*.*"))))
        )
    except Exception as e:
        logger.info(f"Dataset stats might already exist or error occurred: {e}")

    # 2. Generator function wrapper for clean-fid
    def gen_wrapper(z):
        generator.eval()
        with torch.no_grad():
            img = generator(z)
            # Normalize from [-1, 1] to [0, 255] uint8
            img = (img * 0.5 + 0.5) * 255
            img = img.clamp(0, 255).to(torch.uint8)
        return img
        
    logger.info(f"Computing FID with {num_samples} samples...")
    # Calculate FID between generated images and cached real distribution
    score = fid.compute_fid(
        gen=gen_wrapper,
        dataset_name=dataset_name,
        dataset_res=config.image_size,
        num_gen=num_samples,
        dataset_split="custom",
        mode="clean",
        z_dim=config.latent_dim,
        device=device,
        batch_size=min(128, config.batch_size)
    )
    
    logger.info(f"FID Score: {score:.2f}")
    return score
