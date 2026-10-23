"""
Explainability and Feature Attribution Modules
==============================================
Provides modality-level and feature-level interpretability methods:
- Modality ablation and permutation importance
- Attention weight analysis across omics modalities
- Gradient-based feature importance ranking per cancer type
"""

from .modality_importance import (
    compute_modality_ablation_importance,
    compute_modality_permutation_importance,
    analyze_attention_weights,
)
from .feature_importance import (
    compute_feature_attributions,
    get_top_features_per_class,
)

__all__ = [
    "compute_modality_ablation_importance",
    "compute_modality_permutation_importance",
    "analyze_attention_weights",
    "compute_feature_attributions",
    "get_top_features_per_class",
]
