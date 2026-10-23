"""
Per-modality preprocessing pipeline.
Enforces strict anti-leakage: all scaling and feature filtering are fitted on train split only.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ModalityPreprocessor:
    """
    Handles scaling, constant feature filtering, and alignment for a single modality.
    Guarantees no data leakage by fitting exclusively on training samples.
    """
    
    def __init__(
        self,
        modality_name: str,
        normalization: str = "standard",
        variance_threshold: float = 0.0,
    ):
        self.modality_name = modality_name
        self.normalization = normalization.lower()
        self.variance_threshold = variance_threshold
        
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.selected_feature_names: Optional[List[str]] = None
        self.original_feature_count: int = 0
        self.is_fitted: bool = False
        
    def fit(self, X_train: np.ndarray, feature_names: List[str]) -> "ModalityPreprocessor":
        """
        Fit variance filtering and scaler exclusively on training data.
        X_train shape: (n_train_samples, n_features)
        """
        assert X_train.ndim == 2, f"Expected 2D array, got {X_train.shape}"
        assert len(feature_names) == X_train.shape[1], (
            f"Feature count mismatch: {len(feature_names)} vs {X_train.shape[1]}"
        )
        self.original_feature_count = X_train.shape[1]
        
        # 1. Variance filtering
        variances = np.var(X_train, axis=0)
        # Keep features where variance > threshold
        keep_mask = variances > self.variance_threshold
        self.selected_feature_indices = np.where(keep_mask)[0]
        self.selected_feature_names = [feature_names[i] for i in self.selected_feature_indices]
        
        filtered_X_train = X_train[:, self.selected_feature_indices]
        
        # 2. Scaler initialization and fit
        if self.normalization == "standard":
            self.scaler = StandardScaler()
        elif self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalization == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
            
        if self.scaler is not None:
            self.scaler.fit(filtered_X_train)
            
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform a dataset split using train-fitted parameters.
        X shape: (n_samples, n_original_features)
        """
        if not self.is_fitted:
            raise RuntimeError("ModalityPreprocessor must be fitted before calling transform.")
        
        # Filter features
        filtered_X = X[:, self.selected_feature_indices]
        
        # Scale
        if self.scaler is not None:
            transformed = self.scaler.transform(filtered_X)
        else:
            transformed = filtered_X
            
        return transformed.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit on train and return transformed train matrix."""
        self.fit(X_train, feature_names)
        return self.transform(X_train)

    def save(self, filepath: str | Path) -> None:
        """Save fitted preprocessor state to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, filepath: str | Path) -> "ModalityPreprocessor":
        """Load preprocessor from disk."""
        return joblib.load(filepath)
