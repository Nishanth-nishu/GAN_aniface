"""Latent space interpolation."""

import torch
import numpy as np
import torchvision.utils as vutils
import imageio
from pathlib import Path
from tqdm import tqdm


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation.
    
    Unlike standard linear interpolation (lerp), slerp maintains
    constant magnitude during interpolation in high-dimensional spaces,
    which is essential for GAN latent spaces that assume standard normal priors.
    """
    omega = torch.acos(torch.clamp(torch.sum(low/torch.norm(low) * high/torch.norm(high)), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega) / so * high


def create_interpolation_gif(
    generator: torch.nn.Module,
    device: torch.device,
    save_path: str | Path,
    num_frames: int = 60,
    num_transitions: int = 5,
    fps: int = 15
) -> None:
    """Generates an animated GIF transitioning between random latent points.
    
    Args:
        generator: Trained GAN generator.
        device: Computation device.
        save_path: Where to save the animated GIF.
        num_frames: Frames per transition.
        num_transitions: Number of distinct face transitions.
        fps: Frames per second for the output GIF.
    """
    generator.eval()
    
    # Generate anchor points
    anchors = [generator.sample_latent(1, device) for _ in range(num_transitions + 1)]
    # Loop back to start
    anchors[-1] = anchors[0]
    
    frames = []
    
    with torch.no_grad():
        for i in range(num_transitions):
            start_vec = anchors[i]
            end_vec = anchors[i+1]
            
            for f in tqdm(range(num_frames), desc=f"Transition {i+1}/{num_transitions}"):
                alpha = f / float(num_frames)
                interp_vec = slerp(alpha, start_vec, end_vec)
                
                # Generate image
                img_tensor = generator(interp_vec).squeeze(0)
                # Denormalize
                img_tensor = (img_tensor * 0.5) + 0.5
                img_tensor = img_tensor.clamp(0, 1)
                
                # Convert to numpy uint8
                img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                frames.append(img_np)
                
    # Save GIF
    imageio.mimsave(str(save_path), frames, fps=fps)
    generator.train()
