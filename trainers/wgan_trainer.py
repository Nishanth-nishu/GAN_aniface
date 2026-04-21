"""WGAN-GP Trainer."""

import torch
import torchvision.utils as vutils
from typing import Dict, Any

from trainers.base_trainer import BaseTrainer
from losses.adversarial import WassersteinLoss


class WGANGPTrainer(BaseTrainer):
    """Trainer for Wasserstein GAN with Gradient Penalty."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_iterations = getattr(self.config, 'critic_iterations', 5)
        self.gp_weight = getattr(self.config, 'gradient_penalty_weight', 10.0)
        
        # WGAN-GP is very sensitive; often better without mixed precision for discriminator
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        
    def train_step(self, real_imgs: torch.Tensor) -> Dict[str, float]:
        batch_size = real_imgs.size(0)
        
        # ---------------------
        #  Train Critic
        # ---------------------
        d_loss_val = 0.0
        gp_val = 0.0
        
        # Train critic multiple times per generator update
        for _ in range(self.critic_iterations):
            self.opt_d.zero_grad()
            
            z = self.generator.sample_latent(batch_size, self.device)
            fake_imgs = self.generator(z).detach()
            
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                real_pred = self.discriminator(real_imgs)
                fake_pred = self.discriminator(fake_imgs)
                
                # Wasserstein distance
                w_dist = WassersteinLoss.discriminator_loss(real_pred, fake_pred)
                
                # Gradient Penalty (calculate outside autocast to avoid scaling issues)
                gp = WassersteinLoss.gradient_penalty(
                    self.discriminator, real_imgs, fake_imgs, self.device
                )
                
                d_loss = w_dist + self.gp_weight * gp
                
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_d)
            
            d_loss_val += d_loss.item()
            gp_val += gp.item()
            
        d_loss_val /= self.critic_iterations
        gp_val /= self.critic_iterations
        
        # -----------------
        #  Train Generator
        # -----------------
        self.opt_g.zero_grad()
        
        z = self.generator.sample_latent(batch_size, self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            fake_imgs = self.generator(z)
            fake_pred_g = self.discriminator(fake_imgs)
            
            g_loss = WassersteinLoss.generator_loss(fake_pred_g)
            
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()
        
        return {
            "D_loss": d_loss_val,
            "G_loss": g_loss.item(),
            "GP": gp_val,
            "Wasserstein_dist": -w_dist.item() # Expected to be positive
        }
        
    def generate_samples(self) -> None:
        """Generate and save image grid of samples."""
        self.generator.eval()
        with torch.no_grad():
            fake_imgs = self.generator(self.fixed_noise)
            
        fake_imgs = (fake_imgs * 0.5) + 0.5
        save_path = self.config.sample_dir / f"epoch_{self.current_epoch:04d}.png"
        vutils.save_image(fake_imgs, save_path, nrow=8, padding=2, normalize=False)
        self.generator.train()
