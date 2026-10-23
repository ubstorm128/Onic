"""
Multimodal Cancer Dataset implementation.
Supports per-modality tensors, class labels, sample tracking,
and controlled modality missingness simulation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:
        pass


# Standard ordered list of the four omics modalities
MODALITY_KEYS = ["mRNA", "miRNA", "CNV", "Methy"]

# Predefined 15 evaluation conditions (C0 to C14)
# 1 = present, 0 = absent
EVAL_CONDITIONS = {
    "C0":  {"mRNA": 1, "miRNA": 1, "CNV": 1, "Methy": 1, "description": "Complete (reference)"},
    "C1":  {"mRNA": 0, "miRNA": 1, "CNV": 1, "Methy": 1, "description": "mRNA missing"},
    "C2":  {"mRNA": 1, "miRNA": 0, "CNV": 1, "Methy": 1, "description": "miRNA missing"},
    "C3":  {"mRNA": 1, "miRNA": 1, "CNV": 0, "Methy": 1, "description": "CNV missing"},
    "C4":  {"mRNA": 1, "miRNA": 1, "CNV": 1, "Methy": 0, "description": "Methy missing"},
    "C5":  {"mRNA": 0, "miRNA": 0, "CNV": 1, "Methy": 1, "description": "mRNA+miRNA missing"},
    "C6":  {"mRNA": 0, "miRNA": 1, "CNV": 0, "Methy": 1, "description": "mRNA+CNV missing"},
    "C7":  {"mRNA": 0, "miRNA": 1, "CNV": 1, "Methy": 0, "description": "mRNA+Methy missing"},
    "C8":  {"mRNA": 1, "miRNA": 0, "CNV": 0, "Methy": 1, "description": "miRNA+CNV missing"},
    "C9":  {"mRNA": 1, "miRNA": 0, "CNV": 1, "Methy": 0, "description": "miRNA+Methy missing"},
    "C10": {"mRNA": 1, "miRNA": 1, "CNV": 0, "Methy": 0, "description": "CNV+Methy missing"},
    "C11": {"mRNA": 0, "miRNA": 0, "CNV": 0, "Methy": 1, "description": "Only Methy present"},
    "C12": {"mRNA": 0, "miRNA": 0, "CNV": 1, "Methy": 0, "description": "Only CNV present"},
    "C13": {"mRNA": 0, "miRNA": 1, "CNV": 0, "Methy": 0, "description": "Only miRNA present"},
    "C14": {"mRNA": 1, "miRNA": 0, "CNV": 0, "Methy": 0, "description": "Only mRNA present"},
}


class MultimodalCancerDataset(Dataset):
    """
    PyTorch Dataset for multi-omics cancer classification.
    """
    
    def __init__(
        self,
        modality_data: Dict[str, np.ndarray],
        labels: np.ndarray,
        sample_ids: List[str],
        modality_dropout_prob: float = 0.0,
        fixed_condition: Optional[str] = None,
        min_modalities_present: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            modality_data: Dict mapping modality name to numpy array (N, D_m)
            labels: Integer array of class indices (N,)
            sample_ids: List of sample IDs (N,)
            modality_dropout_prob: Training-time probability of dropping each modality independently
            fixed_condition: Predefined evaluation condition key ('C0' through 'C14')
            min_modalities_present: Minimum number of modalities to keep active per sample
            seed: Optional seed for reproducible evaluation
        """
        self.modality_keys = MODALITY_KEYS
        for k in self.modality_keys:
            if k not in modality_data:
                raise KeyError(f"Missing required modality: {k}")
                
        self.num_samples = len(labels)
        for k, arr in modality_data.items():
            assert len(arr) == self.num_samples, f"Sample count mismatch in {k}: {len(arr)} != {self.num_samples}"
            
        self.tensors = {
            k: torch.tensor(modality_data[k], dtype=torch.float32)
            for k in self.modality_keys
        }
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_ids = sample_ids
        
        self.modality_dropout_prob = modality_dropout_prob
        self.fixed_condition = fixed_condition
        self.min_modalities_present = min_modalities_present
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def __len__(self) -> int:
        return self.num_samples

    def get_mask_for_sample(self, idx: int) -> torch.Tensor:
        """
        Generate mask tensor of shape (4,): 1.0 for present, 0.0 for absent.
        """
        # If a fixed condition is specified (e.g. at test time C0-C14)
        if self.fixed_condition is not None:
            cond = EVAL_CONDITIONS[self.fixed_condition]
            mask_vals = [float(cond[k]) for k in self.modality_keys]
            return torch.tensor(mask_vals, dtype=torch.float32)
            
        # If training with modality dropout (Strategy S2 or S3)
        if self.modality_dropout_prob > 0.0:
            probs = self.rng.uniform(0.0, 1.0, size=len(self.modality_keys))
            # 1 = keep, 0 = drop
            mask_vals = (probs >= self.modality_dropout_prob).astype(np.float32)
            
            # Ensure at least min_modalities_present are kept
            if mask_vals.sum() < self.min_modalities_present:
                # Randomly activate one
                activate_idx = self.rng.integers(0, len(self.modality_keys))
                mask_vals[activate_idx] = 1.0
                
            return torch.tensor(mask_vals, dtype=torch.float32)
            
        # Default: all modalities present (complete reference)
        return torch.ones(len(self.modality_keys), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            modality_tensors: Dict mapping modality name to 1D Tensor
            mask: Tensor of shape (4,) where 1=present, 0=absent
            label: Tensor scalar with class index
            sample_id: String sample ID
        """
        sample_tensors = {k: self.tensors[k][idx] for k in self.modality_keys}
        mask = self.get_mask_for_sample(idx)
        label = self.labels[idx]
        sample_id = self.sample_ids[idx]
        
        return sample_tensors, mask, label, sample_id
