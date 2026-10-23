"""
Training and evaluation modules:
- Trainer: Training loop, validation, checkpointing, and early stopping
- Metrics: Macro-F1, Balanced Accuracy, confusion matrices, statistical tests
- Losses: Cross-entropy loss with class weights
"""

from .metrics import compute_metrics, format_metric_summary, wilcoxon_test
from .losses import get_loss_function
from .trainer import ModelTrainer

__all__ = [
    "compute_metrics",
    "format_metric_summary",
    "wilcoxon_test",
    "get_loss_function",
    "ModelTrainer",
]
