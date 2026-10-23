"""
Modality-level importance analysis:
1. Ablation Importance (relative performance drop when modality is masked)
2. Permutation Importance (performance drop when modality values are shuffled across test samples)
3. Attention Weight Analysis (mean attention matrix across modality tokens)
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import MODALITY_KEYS, MultimodalCancerDataset, EVAL_CONDITIONS
from ..training.metrics import compute_metrics


def compute_modality_ablation_importance(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Computes performance drop when each modality is zero-masked or replaced with default token.
    Baseline is complete C0 condition.
    """
    model.eval()
    
    # 1. Baseline C0 evaluation
    c0_preds = []
    c0_targets = []
    with torch.no_grad():
        for mod_tensors, mask, labels, _ in test_loader:
            mod_tensors = {k: v.to(device) for k, v in mod_tensors.items()}
            mask = torch.ones_like(mask).to(device)
            outputs = model(mod_tensors, mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            c0_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            c0_targets.extend(labels.numpy())
            
    c0_metrics = compute_metrics(np.array(c0_targets), np.array(c0_preds))
    baseline_f1 = c0_metrics["macro_f1"]
    
    # 2. Per-modality ablation
    single_missing_conditions = {
        "mRNA": "C1",
        "miRNA": "C2",
        "CNV": "C3",
        "Methy": "C4",
    }
    
    importance = {"baseline_f1": baseline_f1}
    for mod_name, cond_key in single_missing_conditions.items():
        cond_dict = EVAL_CONDITIONS[cond_key]
        cond_mask = torch.tensor([float(cond_dict[k]) for k in MODALITY_KEYS], dtype=torch.float32).to(device)
        
        abl_preds = []
        abl_targets = []
        with torch.no_grad():
            for mod_tensors, _, labels, _ in test_loader:
                mod_tensors = {k: v.to(device) for k, v in mod_tensors.items()}
                b_size = labels.shape[0]
                batch_mask = cond_mask.unsqueeze(0).expand(b_size, -1)
                outputs = model(mod_tensors, batch_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                abl_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                abl_targets.extend(labels.numpy())
                
        abl_metrics = compute_metrics(np.array(abl_targets), np.array(abl_preds))
        abl_f1 = abl_metrics["macro_f1"]
        delta_f1 = baseline_f1 - abl_f1
        importance[f"{mod_name}_ablated_f1"] = round(abl_f1, 4)
        importance[f"{mod_name}_importance_drop"] = round(delta_f1, 4)
        
    return importance


def compute_modality_permutation_importance(
    model: nn.Module,
    test_data: Dict[str, np.ndarray],
    y_test: np.ndarray,
    test_sids: List[str],
    device: str = "cpu",
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes permutation importance for each modality.
    Shuffles all features of a single modality across samples to break cross-sample signal.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    
    # Baseline evaluation
    base_ds = MultimodalCancerDataset(test_data, y_test, test_sids, modality_dropout_prob=0.0)
    base_loader = DataLoader(base_ds, batch_size=64, shuffle=False)
    
    base_preds, base_targets = [], []
    with torch.no_grad():
        for mod_tensors, mask, labels, _ in base_loader:
            mod_tensors = {k: v.to(device) for k, v in mod_tensors.items()}
            mask = mask.to(device)
            outputs = model(mod_tensors, mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            base_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            base_targets.extend(labels.numpy())
            
    base_f1 = compute_metrics(np.array(base_targets), np.array(base_preds))["macro_f1"]
    
    results = {}
    for mod in MODALITY_KEYS:
        drops = []
        for _ in range(n_repeats):
            permuted_data = {k: v.copy() for k, v in test_data.items()}
            # Shuffle rows of the target modality
            perm_indices = rng.permutation(len(y_test))
            permuted_data[mod] = permuted_data[mod][perm_indices]
            
            perm_ds = MultimodalCancerDataset(permuted_data, y_test, test_sids, modality_dropout_prob=0.0)
            perm_loader = DataLoader(perm_ds, batch_size=64, shuffle=False)
            
            p_preds, p_targets = [], []
            with torch.no_grad():
                for mod_tensors, mask, labels, _ in perm_loader:
                    mod_tensors = {k: v.to(device) for k, v in mod_tensors.items()}
                    mask = mask.to(device)
                    outputs = model(mod_tensors, mask)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    p_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    p_targets.extend(labels.numpy())
                    
            perm_f1 = compute_metrics(np.array(p_targets), np.array(p_preds))["macro_f1"]
            drops.append(base_f1 - perm_f1)
            
        results[mod] = {
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
            "baseline_f1": float(base_f1),
        }
        
    return results


def analyze_attention_weights(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extracts and averages cross-attention weights across all test samples.
    Returns:
        mean_attn: (4, 4) numpy array representing average inter-modality attention
    """
    model.eval()
    all_attn = []
    
    with torch.no_grad():
        for mod_tensors, mask, _, _ in test_loader:
            mod_tensors = {k: v.to(device) for k, v in mod_tensors.items()}
            mask = mask.to(device)
            # Request attention weights from forward pass
            logits, attn = model(mod_tensors, mask, return_attention_weights=True)
            if attn is not None:
                all_attn.append(attn.cpu().numpy())
                
    if all_attn:
        stacked = np.concatenate(all_attn, axis=0)  # (N_samples, 4, 4)
        mean_attn = np.mean(stacked, axis=0)
        return mean_attn
    else:
        return np.eye(len(MODALITY_KEYS), dtype=np.float32)
