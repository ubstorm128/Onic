"""
Baseline models:
1. Single-Modality MLPs (mRNA, miRNA, CNV, Methy)
2. Early Fusion (Concatenation) MLP
3. Tabular GBDT Baseline (LightGBM)
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb


class SingleModalityMLP(nn.Module):
    """
    Standard MLP baseline for a single omics modality.
    Architecture:
        Input(K) -> Linear(K, 256) -> BN -> ReLU -> Dropout(0.3)
                 -> Linear(256, 128) -> BN -> ReLU -> Dropout(0.3)
                 -> Linear(128, num_classes)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 4,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (batch_size, input_dim) -> logits: (batch_size, num_classes)"""
        return self.network(x)


class EarlyFusionMLP(nn.Module):
    """
    Early Fusion (Concatenation) baseline model.
    Concatenates all omics modalities at the input layer.
    Architecture:
        Input(K_total) -> Linear(K_total, 512) -> BN -> ReLU -> Dropout(0.3)
                       -> Linear(512, 256) -> BN -> ReLU -> Dropout(0.3)
                       -> Linear(256, 128) -> BN -> ReLU -> Dropout(0.3)
                       -> Linear(128, num_classes)
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 4,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.total_input_dim = sum(modality_dims.values())
        self.num_classes = num_classes
        self.modality_keys = list(modality_dims.keys())
        
        layers = []
        prev_dim = self.total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        modality_tensors: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenates modality tensors in standard order.
        If a modality is missing according to mask, its values are zeroed out.
        """
        # Ensure tensors are concatenated in standard key order
        feature_list = []
        for i, key in enumerate(self.modality_keys):
            feat = modality_tensors[key]
            if mask is not None:
                # mask is (batch_size, num_modalities)
                m = mask[:, i : i + 1]
                feat = feat * m
            feature_list.append(feat)
            
        concatenated = torch.cat(feature_list, dim=-1)
        return self.network(concatenated)


class LightGBMBaseline:
    """
    Conventional tabular gradient-boosted decision tree baseline.
    Evaluates whether deep neural architectures are justified over tabular methods.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            objective="multiclass",
            n_jobs=-1,
            verbosity=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMBaseline":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
