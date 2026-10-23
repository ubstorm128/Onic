"""
Verification Script for Second Pass Audit
========================================
1. Reproduce test metrics from checkpoint: fusion_s1_complete_seed_42_best.pt
2. Inspect exact error counts across all 15 conditions (C0 to C14)
3. Check label alignment and feature names in explainability
4. Verify whether identical metrics are due to dataset properties (discrete sample error counts)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import MultimodalCancerDataset, EVAL_CONDITIONS
from src.models.fusion import MultimodalCrossAttentionClassifier
from src.training.trainer import ModelTrainer
from src.training.metrics import compute_metrics

proc_dir = PROJECT_ROOT / "data" / "processed"
modalities = ["mRNA", "miRNA", "CNV", "Methy"]
test_data = {m: np.load(proc_dir / f"{m}_test.npy") for m in modalities}
y_test = np.load(proc_dir / "labels_test.npy")
manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "splits.csv")
test_sids = manifest[manifest["split"] == "test"]["sample_id"].tolist()
modality_dims = {m: test_data[m].shape[1] for m in modalities}

# Load model
model = MultimodalCrossAttentionClassifier(
    modality_dims=modality_dims,
    encoder_hidden_dim=256,
    latent_dim=128,
    num_heads=4,
    num_layers=1,
    dim_feedforward=256,
    num_classes=4,
    encoder_dropout=0.3,
    fusion_dropout=0.1,
    classifier_dropout=0.3,
    use_modality_type_embeddings=True,
    use_learned_default_token=True,
)

ckpt_path = PROJECT_ROOT / "artifacts" / "models" / "fusion_s1_complete_seed_42_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("--- REPRODUCIBILITY VERIFICATION: EVALUATION OF C0 TO C14 (Seed 42) ---")
print(f"Total test samples: {len(y_test)}")

results_summary = {}

for cond_id, cond_info in EVAL_CONDITIONS.items():
    test_ds = MultimodalCancerDataset(test_data, y_test, test_sids, fixed_condition=cond_id)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_tensors, mask, targets, sids in loader:
            logits, _ = model(batch_tensors, mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    errors = np.where(all_preds != all_targets)[0]
    error_count = len(errors)
    metrics = compute_metrics(all_targets, all_preds)
    
    results_summary[cond_id] = {
        "description": cond_info["description"],
        "error_count": error_count,
        "error_indices": [int(x) for x in errors],
        "accuracy": float(metrics["overall_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
    }
    
    print(f"[{cond_id:4s}] {cond_info['description']:25s} | Errors: {error_count:2d}/340 | Acc: {metrics['overall_accuracy']:.4f} | F1: {metrics['macro_f1']:.4f} | Error Indices: {list(errors)}")

# Save to scratch for verification inspection
scratch_file = PROJECT_ROOT / "scratch_verification_eval.json"
with open(scratch_file, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved verification evaluation to {scratch_file}")
