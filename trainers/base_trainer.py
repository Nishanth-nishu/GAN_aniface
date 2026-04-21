"""Base Trainer class implementation using Template Method pattern."""

import abc
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.checkpointing import CheckpointManager


logger = logging.getLogger(__name__)


class BaseTrainer(abc.ABC):
    """Abstract base class for all trainers.
    
    Implements the Template Method pattern: defines the structure of the 
    training loop and defers specific update steps to subclasses.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        dataloader: DataLoader,
        config: Any,
        device: torch.device,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.dataloader = dataloader
        self.config = config
        self.device = device
        
        self.global_step = 0
        self.current_epoch = 0
        
        # Setup infrastructure
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        if config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(config.log_dir))
        else:
            self.writer = None
            
        # Fixed noise for visual evaluation
        self.fixed_noise = self.generator.sample_latent(64, self.device)
        
    def train(self) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 1. Pre-epoch hook
            self._on_epoch_start()
            
            # 2. Train one epoch
            epoch_metrics = self._train_epoch()
            
            # 3. Post-epoch logging
            self._on_epoch_end(epoch_metrics)
            
            # 4. Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                
            # 5. Generate samples
            if (epoch + 1) % self.config.sample_interval == 0:
                self.generate_samples()
                
            # 6. Evaluate metrics (FID, IS)
            if (epoch + 1) % self.config.eval_interval == 0:
                self.evaluate_metrics()
                
        # Final evaluation
        self.save_checkpoint(is_best=True)
        self.generate_samples()
        
        if self.writer:
            self.writer.close()
            
        logger.info("Training completed.")
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        metrics_sum = {}
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}")
        for batch_idx, data in enumerate(pbar):
            # Move real images to device
            real_imgs = data.to(self.device)
            if self.config.mixed_precision:
                # Use amp autocast context
                with torch.cuda.amp.autocast():
                    step_metrics = self.train_step(real_imgs)
            else:
                step_metrics = self.train_step(real_imgs)
                
            # Accumulate metrics
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            num_batches += 1
            
            # Update progress bar
            self.global_step += 1
            if batch_idx % 10 == 0:
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in step_metrics.items()})
                
                # Log to Tensorboard
                if self.writer:
                    for k, v in step_metrics.items():
                        self.writer.add_scalar(f"Train/{k}", v, self.global_step)
                        
        # Return epoch averages
        return {k: v / num_batches for k, v in metrics_sum.items()}
        
    @abc.abstractmethod
    def train_step(self, real_imgs: torch.Tensor) -> Dict[str, float]:
        """Execute one training step (update D and G).
        
        To be implemented by specific architecture trainers.
        """
        pass
        
    def _on_epoch_start(self) -> None:
        """Hook called before each epoch starts."""
        pass
        
    def _on_epoch_end(self, metrics: Dict[str, float]) -> None:
        """Hook called after an epoch ends."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {self.current_epoch+1}/{self.config.num_epochs} - {metrics_str}")
        
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save a model checkpoint."""
        self.checkpoint_manager.save(
            epoch=self.current_epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            optimizer_g=self.opt_g,
            optimizer_d=self.opt_d,
            is_best=is_best,
            global_step=self.global_step
        )
        
    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        metadata = self.checkpoint_manager.load(
            checkpoint_path=path,
            generator=self.generator,
            discriminator=self.discriminator,
            optimizer_g=self.opt_g,
            optimizer_d=self.opt_d,
            device=self.device
        )
        self.current_epoch = metadata["epoch"] + 1
        self.global_step = metadata.get("global_step", self.current_epoch * len(self.dataloader))
        
    @abc.abstractmethod
    def generate_samples(self) -> None:
        """Generate and save sample images."""
        pass
        
    def evaluate_metrics(self) -> None:
        """Evaluate FID and IS (implemented later)."""
        logger.info("Skipping full evaluation during training to save time.")
