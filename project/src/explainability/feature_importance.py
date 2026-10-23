"""
Feature-level importance and biological attribution:
Computes Gradient-based feature attributions (Input * Gradient)
for each modality, ranking the most discriminative biological markers
per cancer type.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import MODALITY_KEYS


def compute_feature_attributions(
    model: nn.Module,
    X_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    target_class: int,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Computes Input * Gradient attribution for samples belonging to `target_class`.
    
    Returns:
        Dict mapping modality name to mean absolute attribution array (N_features_mod,)
    """
    model.eval()
    
    # Filter samples for the target class
    class_indices = np.where(y == target_class)[0]
    if len(class_indices) == 0:
        return {k: np.zeros(X_dict[k].shape[1], dtype=np.float32) for k in MODALITY_KEYS}
        
    class_X = {k: X_dict[k][class_indices] for k in MODALITY_KEYS}
    num_samples = len(class_indices)
    
    attributions = {k: np.zeros(class_X[k].shape[1], dtype=np.float64) for k in MODALITY_KEYS}
    
    num_batches = int(np.ceil(num_samples / batch_size))
    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, num_samples)
        
        # Prepare inputs with requires_grad=True
        batch_inputs = {}
        for k in MODALITY_KEYS:
            t = torch.tensor(class_X[k][start:end], dtype=torch.float32, device=device, requires_grad=True)
            batch_inputs[k] = t
            
        b_size = end - start
        mask = torch.ones((b_size, len(MODALITY_KEYS)), device=device)
        
        outputs = model(batch_inputs, mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Score for target class
        target_score = logits[:, target_class].sum()
        model.zero_grad()
        target_score.backward()
        
        # Attribution = |input * gradient|
        for k in MODALITY_KEYS:
            grad = batch_inputs[k].grad.detach().cpu().numpy()
            inp = batch_inputs[k].detach().cpu().numpy()
            attr = np.abs(inp * grad)
            attributions[k] += np.sum(attr, axis=0)
            
    # Average across class samples
    for k in MODALITY_KEYS:
        attributions[k] = (attributions[k] / num_samples).astype(np.float32)
        
    return attributions


def get_top_features_per_class(
    attributions_per_class: Dict[int, Dict[str, np.ndarray]],
    feature_names_dict: Dict[str, List[str]],
    top_k: int = 10,
    class_names: List[str] = ["BRCA", "HNSC", "LGG", "THCA"],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Ranks the top-K features per cancer class and modality.
    """
    results = {}
    for c_idx, c_name in enumerate(class_names):
        results[c_name] = {}
        for mod in MODALITY_KEYS:
            attrs = attributions_per_class[c_idx][mod]
            names = feature_names_dict[mod]
            top_indices = np.argsort(attrs)[::-1][:top_k]
            
            top_list = [
                {
                    "rank": r + 1,
                    "feature": names[idx],
                    "attribution_score": float(attrs[idx]),
                }
                for r, idx in enumerate(top_indices)
            ]
            results[c_name][mod] = top_list
            
    return results
