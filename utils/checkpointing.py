"""
Model checkpointing utilities.

Handles saving, loading, and managing model checkpoints with
metadata tracking for training state recovery.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with metadata and best-model tracking.

    Follows Single Responsibility Principle: only handles persistence
    of model/optimizer state and training metadata.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        max_checkpoints: Maximum checkpoints to keep (oldest deleted first).
    """

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._checkpoint_history: list = []

    def save(
        self,
        epoch: int,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        **extra_state: Any,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            generator: Generator model.
            discriminator: Discriminator model.
            optimizer_g: Generator optimizer.
            optimizer_d: Discriminator optimizer.
            metrics: Optional evaluation metrics dict.
            is_best: If True, also saves as 'best_model.pt'.
            **extra_state: Additional state to save (e.g., scaler).

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "metrics": metrics or {},
        }
        checkpoint.update(extra_state)

        # Save numbered checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # Track and prune old checkpoints
        self._checkpoint_history.append(ckpt_path)
        self._prune_checkpoints()

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Save latest (for easy resume)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save metadata
        self._save_metadata(epoch, metrics)

        return ckpt_path

    def load(
        self,
        checkpoint_path: str,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: Optional[torch.optim.Optimizer] = None,
        optimizer_d: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            generator: Generator model to load state into.
            discriminator: Discriminator model to load state into.
            optimizer_g: Optional generator optimizer to restore.
            optimizer_d: Optional discriminator optimizer to restore.
            device: Device to map tensors to.

        Returns:
            Dictionary with checkpoint metadata (epoch, metrics, etc.).
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
            optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])

        if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
            optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}: "
            f"{ckpt_path}"
        )
        return checkpoint

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        while len(self._checkpoint_history) > self.max_checkpoints:
            old_ckpt = self._checkpoint_history.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                logger.debug(f"Pruned old checkpoint: {old_ckpt}")

    def _save_metadata(
        self, epoch: int, metrics: Optional[Dict[str, float]]
    ) -> None:
        """Save training metadata as JSON for inspection."""
        meta_path = self.checkpoint_dir / "training_metadata.json"
        metadata = {"last_epoch": epoch, "last_metrics": metrics or {}}
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
