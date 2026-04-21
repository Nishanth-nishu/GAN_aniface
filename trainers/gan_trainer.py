"""Standard GAN trainer (used for DCGAN, SAGAN, USE-CMHSA)."""

import torch
import torchvision.utils as vutils
from pathlib import Path
from typing import Dict, Any

from trainers.base_trainer import BaseTrainer
from losses.adversarial import AdversarialLoss, HingeLoss


class GANTrainer(BaseTrainer):
    """Trainer for standard adversarial networks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Loss function selection
        if self.config.model_type == "sagan":
            self.criterion_d = HingeLoss.discriminator_loss
            self.criterion_g = HingeLoss.generator_loss
            self.loss_type = "hinge"
        else:
            bce = AdversarialLoss(getattr(self.config, 'label_smoothing', 0.0))
            self.criterion_d = bce
            self.criterion_g = bce
            self.loss_type = "bce"
            
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        
    def train_step(self, real_imgs: torch.Tensor) -> Dict[str, float]:
        batch_size = real_imgs.size(0)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.opt_d.zero_grad()
        
        # Real images
        real_pred = self.discriminator(real_imgs)
        
        # Fake images
        z = self.generator.sample_latent(batch_size, self.device)
        fake_imgs = self.generator(z)
        fake_pred = self.discriminator(fake_imgs.detach())
        
        # Loss computation
        if self.loss_type == "hinge":
            d_loss = self.criterion_d(real_pred, fake_pred)
        else: # BCE
            d_real_loss = self.criterion_d(real_pred, True)
            d_fake_loss = self.criterion_d(fake_pred, False)
            d_loss = d_real_loss + d_fake_loss
            
        # Optimization step
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        
        # -----------------
        #  Train Generator
        # -----------------
        self.opt_g.zero_grad()
        
        # We need fresh predictions for the updated generator
        fake_pred_g = self.discriminator(fake_imgs)
        
        # Loss computation
        if self.loss_type == "hinge":
            g_loss = self.criterion_g(fake_pred_g)
        else: # BCE
            # Generator wants discriminator to think fakes are real
            g_loss = self.criterion_g(fake_pred_g, True)
            
        # Optimization step
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()
        
        return {
            "D_loss": d_loss.item(),
            "G_loss": g_loss.item(),
            "D_real": real_pred.mean().item(),
            "D_fake": fake_pred.mean().item()
        }
        
    def generate_samples(self) -> None:
        """Generate and save image grid of samples."""
        self.generator.eval()
        with torch.no_grad():
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    fake_imgs = self.generator(self.fixed_noise)
            else:
                fake_imgs = self.generator(self.fixed_noise)
                
        # Denormalize from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs * 0.5) + 0.5
        
        # Save image grid
        save_path = self.config.sample_dir / f"epoch_{self.current_epoch:04d}.png"
        vutils.save_image(fake_imgs, save_path, nrow=8, padding=2, normalize=False)
        self.generator.train()
