"""
Device management utilities.

Handles GPU/CPU detection, memory reporting, and device placement
for tensors and models.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "cuda") -> torch.device:
    """Get the best available computation device.

    Args:
        preferred: Preferred device string ('cuda', 'cpu', 'cuda:0', etc.).

    Returns:
        torch.device configured for the best available hardware.
    """
    if preferred.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(preferred)
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        gpu_memory_gb = gpu_memory / (1024 ** 3)
        logger.info(
            f"Using GPU: {gpu_name} ({gpu_memory_gb:.1f} GB VRAM)"
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no CUDA GPU available)")

    return device


def log_gpu_memory(device: torch.device) -> None:
    """Log current GPU memory usage."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        logger.info(
            f"GPU Memory: {allocated:.2f} GB allocated, "
            f"{reserved:.2f} GB reserved"
        )
