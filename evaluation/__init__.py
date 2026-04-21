"""Evaluation package init."""
from evaluation.fid import compute_fid
from evaluation.evaluator import Evaluator

__all__ = ["compute_fid", "Evaluator"]
