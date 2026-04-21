"""Unified Evaluator for all metrics."""

import logging
from pathlib import Path
from typing import Dict, List, Any
import json

import torch

from evaluation.fid import compute_fid

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs all evaluation metrics and generates reports."""
    
    def __init__(self, config: Any, generator: torch.nn.Module, device: torch.device):
        self.config = config
        self.generator = generator
        self.device = device
        self.results: Dict[str, float] = {}
        
    def evaluate_all(self, metrics: List[str] = ["fid"]) -> Dict[str, float]:
        """Run requested evaluation metrics.
        
        Args:
            metrics: List of metrics to compute (e.g. ['fid', 'is']).
            
        Returns:
            Dictionary mapping metric names to scores.
        """
        if "fid" in metrics:
            try:
                fid_score = compute_fid(
                    self.config, 
                    self.generator, 
                    num_samples=self.config.num_eval_samples,
                    device=self.device
                )
                self.results["fid"] = fid_score
            except Exception as e:
                logger.error(f"FID computation failed: {e}")
                
        # Space for other metrics (IS, LPIPS)
        if "is" in metrics:
            logger.warning("Inception Score not yet fully implemented. Skipping.")
            self.results["is"] = -1.0
            
        self._save_results()
        return self.results
        
    def _save_results(self) -> None:
        """Save results to JSON."""
        metrics_file = self.config.metrics_dir / f"eval_results_{self.config.model_type}.json"
        
        with open(metrics_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Saved evaluation results to {metrics_file}")
        
    def print_summary(self) -> None:
        """Print formatted summary."""
        print("\n" + "="*40)
        print(f" EVALUATION SUMMARY: {self.config.model_type.upper()}")
        print("="*40)
        for metric, value in self.results.items():
            print(f" * {metric.upper():<10} : {value:.4f}")
        print("="*40 + "\n")
