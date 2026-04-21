"""Visualization tools for GAN training and inference."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    g_losses: list, 
    d_losses: list, 
    save_path: str | Path,
    title: str = "Generator and Discriminator Loss During Training"
) -> None:
    """Plot and save training loss curves.
    
    Args:
        g_losses: List of generator losses.
        d_losses: List of discriminator losses.
        save_path: Where to save the plot image.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(g_losses, label="Generator", alpha=0.8)
    plt.plot(d_losses, label="Discriminator", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def plot_metrics_history(
    metrics_history: dict, 
    save_path: str | Path
) -> None:
    """Plot and save evaluation metrics over time.
    
    Args:
        metrics_history: Dictionary mapping metric name to list of values.
        save_path: Where to save the plot image.
    """
    if not metrics_history:
        return
        
    num_metrics = len(metrics_history)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
        
    for ax, (metric_name, values) in zip(axes, metrics_history.items()):
        ax.plot(values, marker='o', linestyle='-', linewidth=2)
        ax.set_title(f"{metric_name.upper()} Score over Training")
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()
