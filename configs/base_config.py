"""
Base configuration dataclass for all GAN architectures.

Provides a unified configuration interface following the Open/Closed Principle:
subclasses extend this base without modifying it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class BaseConfig:
    """Base configuration shared across all GAN architectures.

    Attributes:
        experiment_name: Unique name for the training run.
        model_type: Architecture type identifier.
        image_size: Target image resolution (square).
        channels: Number of image channels (3 for RGB).
        latent_dim: Dimensionality of the latent noise vector z.
        batch_size: Training batch size.
        num_epochs: Total training epochs.
        learning_rate_g: Generator learning rate.
        learning_rate_d: Discriminator learning rate.
        beta1: Adam optimizer beta1 parameter.
        beta2: Adam optimizer beta2 parameter.
        num_workers: DataLoader worker threads.
        device: Computation device ('cuda' or 'cpu').
        seed: Random seed for reproducibility.
        data_dir: Path to preprocessed dataset.
        output_dir: Root output directory.
        checkpoint_interval: Save checkpoint every N epochs.
        sample_interval: Generate sample images every N epochs.
        eval_interval: Run evaluation metrics every N epochs.
        num_eval_samples: Number of samples for FID/IS computation.
        resume_checkpoint: Optional path to resume training.
        use_tensorboard: Enable TensorBoard logging.
        mixed_precision: Enable AMP mixed-precision training.
    """

    # --- Experiment ---
    experiment_name: str = "anime_gan"
    model_type: str = "base"

    # --- Image ---
    image_size: int = 64
    channels: int = 3

    # --- Model ---
    latent_dim: int = 128
    g_filters: int = 64
    d_filters: int = 64

    # --- Training ---
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 4

    # --- Device ---
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True

    # --- Data ---
    data_dir: str = "/scratch/nishanth.r/gan_proj/data_processed"
    output_dir: str = "/scratch/nishanth.r/gan_proj/outputs"

    # --- Checkpointing ---
    checkpoint_interval: int = 10
    sample_interval: int = 5
    eval_interval: int = 25
    num_eval_samples: int = 10000
    resume_checkpoint: Optional[str] = None

    # --- Logging ---
    use_tensorboard: bool = True

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for model checkpoints."""
        return Path(self.output_dir) / self.experiment_name / "checkpoints"

    @property
    def sample_dir(self) -> Path:
        """Directory for generated sample images."""
        return Path(self.output_dir) / self.experiment_name / "samples"

    @property
    def metrics_dir(self) -> Path:
        """Directory for evaluation metrics."""
        return Path(self.output_dir) / self.experiment_name / "metrics"

    @property
    def log_dir(self) -> Path:
        """Directory for TensorBoard logs."""
        return Path(self.output_dir) / self.experiment_name / "logs"

    def create_directories(self) -> None:
        """Create all output directories."""
        for dir_path in [
            self.checkpoint_dir,
            self.sample_dir,
            self.metrics_dir,
            self.log_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()}
