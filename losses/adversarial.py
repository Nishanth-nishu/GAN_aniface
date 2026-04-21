"""Loss functions for GAN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialLoss(nn.Module):
    """Standard non-saturating adversarial loss (BCEWithLogits).
    
    Used in DCGAN and USE-CMHSA-GAN.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing
        
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target_val = 1.0 - self.label_smoothing
        else:
            target_val = 0.0
            
        return torch.full_like(prediction, target_val, requires_grad=False)
        
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss_fn(prediction, target_tensor)


class HingeLoss:
    """Hinge loss for adversarial training.
    
    Used in SAGAN.
    """
    @staticmethod
    def discriminator_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        loss_real = F.relu(1.0 - real_pred).mean()
        loss_fake = F.relu(1.0 + fake_pred).mean()
        return loss_real + loss_fake
        
    @staticmethod
    def generator_loss(fake_pred: torch.Tensor) -> torch.Tensor:
        return -fake_pred.mean()


class WassersteinLoss:
    """Wasserstein loss with gradient penalty.
    
    Used in WGAN-GP.
    """
    @staticmethod
    def discriminator_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return fake_pred.mean() - real_pred.mean()
        
    @staticmethod
    def generator_loss(fake_pred: torch.Tensor) -> torch.Tensor:
        return -fake_pred.mean()
        
    @staticmethod
    def gradient_penalty(
        discriminator: nn.Module, 
        real_samples: torch.Tensor, 
        fake_samples: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """Calculates the gradient penalty.
        
        Args:
            discriminator: The critic network.
            real_samples: Real images (batch_size, C, H, W).
            fake_samples: Generated images (batch_size, C, H, W).
            device: Computation device.
            
        Returns:
            Gradient penalty tensor.
        """
        batch_size = real_samples.size(0)
        
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((batch_size, 1, 1, 1), device=device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1.0 - alpha) * fake_samples)).requires_grad_(True)
        
        # Calculate discriminator output on interpolates
        d_interpolates = discriminator(interpolates)
        
        # Get gradients of d_interpolates w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1.0) ** 2).mean()
        
        return penalty
