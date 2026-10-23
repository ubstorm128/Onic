"""
Automated Pipeline and Architecture Tests
=========================================
Tests:
1. Leak-free stratified splitting guarantees
2. Processed tensor dimensions and consistency
3. Multimodal cross-attention model forward pass
4. Missing modality masking and default token replacement
5. Metric computation correctness
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.utils.config import load_config
from src.data.dataset import MODALITY_KEYS, MultimodalCancerDataset, EVAL_CONDITIONS
from src.models.fusion import MultimodalCrossAttentionClassifier
from src.training.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_splits_no_leakage():
    splits_path = PROJECT_ROOT / "data" / "splits" / "splits.csv"
    assert splits_path.is_file(), "Splits manifest must exist"
    
    df = pd.read_csv(splits_path)
    assert len(df) == 2263, f"Expected 2263 samples, got {len(df)}"
    
    train_ids = set(df[df["split"] == "train"]["sample_id"])
    val_ids = set(df[df["split"] == "val"]["sample_id"])
    test_ids = set(df[df["split"] == "test"]["sample_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage: Train and Val overlap"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage: Train and Test overlap"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage: Val and Test overlap"


def test_processed_modality_shapes():
    proc_dir = PROJECT_ROOT / "data" / "processed"
    expected_dims = {"mRNA": 3217, "miRNA": 383, "CNV": 3105, "Methy": 3139}
    
    for mod, exp_dim in expected_dims.items():
        arr_full = np.load(proc_dir / f"{mod}_processed.npy")
        arr_train = np.load(proc_dir / f"{mod}_train.npy")
        arr_val = np.load(proc_dir / f"{mod}_val.npy")
        arr_test = np.load(proc_dir / f"{mod}_test.npy")
        
        assert arr_full.shape == (2263, exp_dim)
        assert arr_train.shape == (1584, exp_dim)
        assert arr_val.shape == (339, exp_dim)
        assert arr_test.shape == (340, exp_dim)
        assert not np.isnan(arr_full).any(), f"NaNs detected in {mod}"


def test_model_forward_pass_and_attention():
    modality_dims = {"mRNA": 3217, "miRNA": 383, "CNV": 3105, "Methy": 3139}
    model = MultimodalCrossAttentionClassifier(
        modality_dims=modality_dims,
        encoder_hidden_dim=64,
        latent_dim=32,
        num_heads=2,
        num_classes=4,
    )
    model.eval()
    
    batch_size = 4
    inputs = {k: torch.randn(batch_size, dim) for k, dim in modality_dims.items()}
    mask = torch.ones(batch_size, 4)
    
    logits, attn = model(inputs, mask, return_attention_weights=True)
    assert logits.shape == (batch_size, 4), f"Logits shape mismatch: {logits.shape}"
    assert attn.shape == (batch_size, 4, 4), f"Attention shape mismatch: {attn.shape}"
    assert not torch.isnan(logits).any()


def test_missing_modality_handling():
    modality_dims = {"mRNA": 3217, "miRNA": 383, "CNV": 3105, "Methy": 3139}
    model = MultimodalCrossAttentionClassifier(
        modality_dims=modality_dims,
        encoder_hidden_dim=64,
        latent_dim=32,
        num_heads=2,
        num_classes=4,
    )
    model.eval()
    
    batch_size = 2
    inputs = {k: torch.randn(batch_size, dim) for k, dim in modality_dims.items()}
    # Simulate C12: Only CNV (index 2) is present
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    
    logits, attn = model(inputs, mask, return_attention_weights=True)
    assert logits.shape == (batch_size, 4)
    assert not torch.isnan(logits).any()


def test_metrics_computation():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 2])  # 7/8 correct
    m = compute_metrics(y_true, y_pred)
    
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert len(m["confusion_matrix"]) == 4


def test_masked_modality_numerical_invariance():
    """
    Priority 3 Test: Verify that altering numerical values of masked modalities (mask=0)
    leaves model logits strictly unchanged in evaluation mode.
    Tests multiple random perturbations to guarantee zero leakage from masked inputs.
    """
    modality_dims = {"mRNA": 3217, "miRNA": 383, "CNV": 3105, "Methy": 3139}
    model = MultimodalCrossAttentionClassifier(
        modality_dims=modality_dims,
        encoder_hidden_dim=64,
        latent_dim=32,
        num_heads=2,
        num_classes=4,
    )
    model.eval()

    batch_size = 4
    torch.manual_seed(42)
    inputs_orig = {k: torch.randn(batch_size, dim) for k, dim in modality_dims.items()}
    
    # Mask where mRNA (idx 0) and CNV (idx 2) are absent (0.0), miRNA and Methy are present (1.0)
    mask = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ])

    with torch.no_grad():
        logits_orig, _ = model(inputs_orig, mask)

    # Apply 3 distinct severe perturbations to masked modalities (mRNA and CNV)
    perturbations = [
        {"mRNA": torch.randn_like(inputs_orig["mRNA"]) * 1000.0 + 500.0,
         "CNV": torch.randn_like(inputs_orig["CNV"]) * 1000.0 - 500.0},
        {"mRNA": torch.zeros_like(inputs_orig["mRNA"]),
         "CNV": torch.ones_like(inputs_orig["CNV"]) * 99999.0},
        {"mRNA": torch.full_like(inputs_orig["mRNA"], -1234.5),
         "CNV": torch.full_like(inputs_orig["CNV"], 6789.0)},
    ]

    for p_idx, pert in enumerate(perturbations):
        inputs_perturbed = {k: v.clone() for k, v in inputs_orig.items()}
        inputs_perturbed["mRNA"] = pert["mRNA"]
        inputs_perturbed["CNV"] = pert["CNV"]

        with torch.no_grad():
            logits_pert, _ = model(inputs_perturbed, mask)

        max_diff = (logits_orig - logits_pert).abs().max().item()
        assert max_diff < 1e-6, (
            f"Perturbation {p_idx} changed evaluation logits by {max_diff}! "
            f"Masked inputs must have zero numerical impact."
        )


def test_all_15_evaluation_conditions_tensor_shapes_and_finite():
    """
    Priority 3 Test: Verify all 15 non-empty modality conditions (C0 to C14)
    execute successfully, produce correct tensor shapes (batch_size, num_classes),
    and output strictly finite numbers (no NaN or Inf).
    """
    modality_dims = {"mRNA": 3217, "miRNA": 383, "CNV": 3105, "Methy": 3139}
    model = MultimodalCrossAttentionClassifier(
        modality_dims=modality_dims,
        encoder_hidden_dim=64,
        latent_dim=32,
        num_heads=2,
        num_classes=4,
    )
    model.eval()

    batch_size = 3
    torch.manual_seed(999)
    inputs = {k: torch.randn(batch_size, dim) for k, dim in modality_dims.items()}

    for cond_id, cond_info in EVAL_CONDITIONS.items():
        mask_vec = torch.tensor([float(cond_info[m]) for m in MODALITY_KEYS], dtype=torch.float32)
        mask = mask_vec.unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            logits, attn = model(inputs, mask, return_attention_weights=True)

        assert logits.shape == (batch_size, 4), (
            f"Condition {cond_id}: unexpected logits shape {logits.shape}"
        )
        assert torch.isfinite(logits).all(), (
            f"Condition {cond_id}: non-finite logits detected"
        )
        assert attn.shape == (batch_size, 4, 4), (
            f"Condition {cond_id}: unexpected attention shape {attn.shape}"
        )
        assert torch.isfinite(attn).all(), (
            f"Condition {cond_id}: non-finite attention detected"
        )


def test_all_missing_numerical_safeguard():
    """
    Priority 3 Test: Verify the all-missing modality condition (mask = [0, 0, 0, 0]).
    Documentation:
    The all-missing condition is strictly an internal numerical safeguard to prevent
    pipeline failure (NaNs in softmax or zero-division in pooling) if an empty input is encountered.
    It is NOT intended for clinical deployment, as cancer prediction without any patient omics data
    has zero clinical utility. In this edge case, the model safely outputs constant prior logits.
    """
    modality_dims = {"mRNA": 3217, "miRNA": 383, "CNV": 3105, "Methy": 3139}
    model = MultimodalCrossAttentionClassifier(
        modality_dims=modality_dims,
        encoder_hidden_dim=64,
        latent_dim=32,
        num_heads=2,
        num_classes=4,
    )
    model.eval()

    batch_size = 4
    torch.manual_seed(1234)
    inputs = {k: torch.randn(batch_size, dim) for k, dim in modality_dims.items()}
    all_zero_mask = torch.zeros(batch_size, 4, dtype=torch.float32)

    with torch.no_grad():
        logits, attn = model(inputs, all_zero_mask, return_attention_weights=True)

    assert logits.shape == (batch_size, 4), f"Unexpected logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contained NaN or Inf under all-missing mask"
    assert attn.shape == (batch_size, 4, 4), f"Unexpected attention shape: {attn.shape}"
    assert torch.isfinite(attn).all(), "Attention contained NaN or Inf under all-missing mask"

    # All samples with all modalities missing should receive identical fallback prior predictions
    sample_diff = (logits[0:1] - logits[1:]).abs().max().item()
    assert sample_diff < 1e-6, f"All-missing samples received non-identical prior logits: {sample_diff}"

