"""
Loss functions with optional class-weighting for class imbalance.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def get_loss_function(
    class_weights: Optional[np.ndarray | torch.Tensor] = None,
    device: str = "cpu",
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """
    Returns standard or class-weighted CrossEntropyLoss.
    """
    weights_tensor = None
    if class_weights is not None:
        if isinstance(class_weights, np.ndarray):
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        elif isinstance(class_weights, torch.Tensor):
            weights_tensor = class_weights.to(device)
            
    return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
