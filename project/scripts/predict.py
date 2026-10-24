"""
CLI Standalone Inference Tool for Multimodal Cancer Classification
==================================================================
Runs the Multimodal Cross-Attention Model directly without requiring any frontend.
Supports:
1. Simulated patient omics profiles or real processed test samples
2. Arbitrary missing modality combinations (C0 to C14 or custom masks)
3. Cancer class prediction (BRCA, HNSC, LGG, THCA) with confidence breakdown
4. Dynamic cross-attention routing inspection across modalities
5. JSON export of predictions and attention weights

Usage Examples:
    # 1. Run inference on a simulated sample under complete modalities (C0):
    python scripts/predict.py --simulate

    # 2. Run inference with specific missing modalities (e.g., CNV and Methy missing):
    python scripts/predict.py --simulate --missing CNV Methy

    # 3. Run inference using a predefined condition code (C1 to C14):
    python scripts/predict.py --simulate --condition C12

    # 4. Run inference on a real test sample (if data/processed exists):
    python scripts/predict.py --sample-index 0 --condition C3

    # 5. Export prediction to JSON:
    python scripts/predict.py --simulate --save-json output.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch is required. Please install torch: pip install torch")
    sys.exit(1)

from src.models.fusion import MultimodalCrossAttentionClassifier
from src.data.dataset import MODALITY_KEYS, EVAL_CONDITIONS

# TCGA Cancer Classification Label Mapping
CANCER_CLASSES = {
    0: ("BRCA", "Breast Invasive Carcinoma"),
    1: ("HNSC", "Head and Neck Squamous Cell Carcinoma"),
    2: ("LGG",  "Brain Lower Grade Glioma"),
    3: ("THCA", "Thyroid Carcinoma"),
}

DEFAULT_MODALITY_DIMS = {
    "mRNA": 3217,
    "miRNA": 383,
    "CNV": 3105,
    "Methy": 3139,
}


def create_model(checkpoint_path: Optional[Path] = None, seed: int = 42) -> MultimodalCrossAttentionClassifier:
    """Instantiate and load the Multimodal Cross-Attention Classifier."""
    torch.manual_seed(seed)
    model = MultimodalCrossAttentionClassifier(
        modality_dims=DEFAULT_MODALITY_DIMS,
        encoder_hidden_dim=256,
        latent_dim=128,
        num_heads=4,
        num_layers=1,
        dim_feedforward=256,
        num_classes=4,
        encoder_dropout=0.0,
        fusion_dropout=0.0,
        classifier_dropout=0.0,
        use_modality_type_embeddings=True,
        use_learned_default_token=True,
    )
    
    if checkpoint_path and checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded model checkpoint from: {checkpoint_path}")
    else:
        # Check default checkpoint location
        default_ckpt = PROJECT_ROOT / "artifacts" / "models" / "fusion_s1_complete_seed_42_best.pt"
        if default_ckpt.is_file():
            ckpt = torch.load(default_ckpt, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded default checkpoint: {default_ckpt.name}")
        else:
            print("[INFO] No saved checkpoint file found; initialized reference architecture weights.")
            
    model.eval()
    return model


def get_mask(
    condition: Optional[str] = None,
    missing: Optional[List[str]] = None,
    available: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, str]:
    """Build binary presence mask (1=present, 0=absent)."""
    mask_dict = {m: 1.0 for m in MODALITY_KEYS}
    description = "Complete Modalities"
    
    if condition:
        cond_upper = condition.upper()
        if cond_upper in EVAL_CONDITIONS:
            cond_info = EVAL_CONDITIONS[cond_upper]
            for m in MODALITY_KEYS:
                mask_dict[m] = float(cond_info[m])
            description = f"{cond_upper}: {cond_info['description']}"
        else:
            print(f"[WARNING] Unknown condition '{condition}'. Valid conditions: C0-C14. Falling back to C0.")
    elif missing:
        missing_set = set(missing)
        for m in MODALITY_KEYS:
            if m in missing_set:
                mask_dict[m] = 0.0
        description = f"Custom: Missing {', '.join(sorted(missing_set))}"
    elif available:
        avail_set = set(available)
        for m in MODALITY_KEYS:
            mask_dict[m] = 1.0 if m in avail_set else 0.0
        description = f"Custom: Available {', '.join(sorted(avail_set))}"
        
    mask_tensor = torch.tensor([[mask_dict[m] for m in MODALITY_KEYS]], dtype=torch.float32)
    return mask_tensor, description


def generate_simulated_inputs(seed: int = 42) -> Dict[str, torch.Tensor]:
    """Generate realistic standard-normal synthetic inputs for demonstration."""
    torch.manual_seed(seed)
    inputs = {}
    for mod, dim in DEFAULT_MODALITY_DIMS.items():
        if mod == "Methy":
            # Methy is min-max scaled in [0, 1]
            inputs[mod] = torch.rand(1, dim)
        else:
            # mRNA, miRNA, CNV are z-score standardized
            inputs[mod] = torch.randn(1, dim)
    return inputs


def load_test_sample(sample_idx: int) -> Tuple[Dict[str, torch.Tensor], Optional[int], str]:
    """Load an actual test sample from data/processed if available."""
    proc_dir = PROJECT_ROOT / "data" / "processed"
    splits_file = PROJECT_ROOT / "data" / "splits" / "splits.csv"
    
    if not (proc_dir / "mRNA_test.npy").is_file():
        raise FileNotFoundError(
            f"Processed data not found at {proc_dir}. Use '--simulate' to run inference with simulated data."
        )
        
    inputs = {}
    for mod in MODALITY_KEYS:
        arr = np.load(proc_dir / f"{mod}_test.npy")
        if sample_idx >= len(arr):
            raise IndexError(f"sample_idx {sample_idx} exceeds test set size ({len(arr)})")
        inputs[mod] = torch.from_numpy(arr[sample_idx:sample_idx+1]).float()
        
    y_test = np.load(proc_dir / "labels_test.npy")
    true_label = int(y_test[sample_idx])
    
    sample_id = f"Test_Sample_{sample_idx}"
    if splits_file.is_file():
        df_splits = pd.read_csv(splits_file)
        test_ids = df_splits[df_splits["split"] == "test"]["sample_id"].tolist()
        if sample_idx < len(test_ids):
            sample_id = test_ids[sample_idx]
            
    return inputs, true_label, sample_id


def format_progress_bar(val: float, total: float = 1.0, length: int = 24) -> str:
    """Format visual text progress bar."""
    filled = int(round(length * (val / total)))
    return f"[{'=' * filled}{' ' * (length - filled)}]"


def print_prediction_dashboard(
    inputs: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    condition_desc: str,
    logits: torch.Tensor,
    attn_weights: Optional[torch.Tensor],
    sample_id: str = "Sample-01",
    true_label: Optional[int] = None,
):
    """Print a clean, structured CLI dashboard for the inference results."""
    probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_code, pred_name = CANCER_CLASSES[pred_idx]
    pred_conf = probs[pred_idx] * 100.0
    
    print("\n" + "=" * 70)
    print("      MULTIMODAL AI CANCER CLASSIFIER - INFERENCE RESULTS      ")
    print("=" * 70)
    print(f" Sample ID   : {sample_id}")
    print(f" Condition   : {condition_desc}")
    if true_label is not None:
        true_code, true_name = CANCER_CLASSES[true_label]
        match_str = "CORRECT [PASS]" if true_label == pred_idx else "MISMATCH [FAIL]"
        print(f" Ground Truth: {true_code} ({true_name}) -> {match_str}")
    print("-" * 70)
    
    # 1. Modality Availability Status
    print(" MODALITY INPUT STATUS:")
    mask_vals = mask.squeeze(0).tolist()
    for i, mod in enumerate(MODALITY_KEYS):
        is_present = (mask_vals[i] > 0.5)
        status_icon = "[PRESENT]" if is_present else "[MISSING]"
        handling_note = f"Active ({DEFAULT_MODALITY_DIMS[mod]} features)" if is_present else "Replaced with Learned Default Token"
        print(f"   {mod:6s} : {status_icon:9s} -> {handling_note}")
    print("-" * 70)
    
    # 2. Prediction Breakdown
    print(" PREDICTION PROBABILITIES:")
    for idx, (code, full_name) in CANCER_CLASSES.items():
        p = probs[idx]
        bar = format_progress_bar(p)
        marker = " <-- PREDICTED" if idx == pred_idx else ""
        print(f"   {code:5s} ({full_name:38s}) : {p*100:5.2f}% {bar}{marker}")
        
    print("-" * 70)
    print(f" FINAL DIAGNOSIS: {pred_code} - {pred_name}")
    print(f" CONFIDENCE      : {pred_conf:.2f}%")
    print("-" * 70)
    
    # 3. Cross-Attention Matrix
    if attn_weights is not None:
        print(" CROSS-ATTENTION ROUTING MATRIX (Inter-Modality Attention Weights):")
        attn_matrix = attn_weights.squeeze(0).cpu().numpy()
        header = "   Query \\ Key | " + " | ".join(f"{m:>7s}" for m in MODALITY_KEYS) + " |"
        print(header)
        print("   " + "-" * (len(header) - 3))
        for i, q_mod in enumerate(MODALITY_KEYS):
            row_str = f"   {q_mod:>11s} | "
            for j, k_mod in enumerate(MODALITY_KEYS):
                val = attn_matrix[i, j]
                row_str += f"{val:7.4f} | "
            if mask_vals[i] == 0:
                row_str += " (Masked Input)"
            print(row_str)
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Multimodal Cancer Classification Inference without a frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simulate", action="store_true", default=False,
        help="Run inference on a synthetic patient profile (useful when raw files are not on disk)."
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Predefined evaluation condition: C0 (complete), C1-C4 (single missing), C5-C10 (two missing), C11-C14 (single present)."
    )
    parser.add_argument(
        "--missing", nargs="+", choices=MODALITY_KEYS, default=None,
        help="Specify which modalities to mask out (e.g. --missing mRNA CNV)."
    )
    parser.add_argument(
        "--available", nargs="+", choices=MODALITY_KEYS, default=None,
        help="Specify which modalities are available (e.g. --available mRNA miRNA)."
    )
    parser.add_argument(
        "--sample-index", type=int, default=None,
        help="Test sample index to load from data/processed (0 to 339)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to saved model checkpoint (.pt)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for simulated profile generation or weight initialization."
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Optional path to save prediction results as JSON."
    )
    
    args = parser.parse_args()
    
    # 1. Determine Inputs
    sample_id = "Simulated-Patient-001"
    true_label = None
    
    if args.sample_index is not None:
        try:
            inputs, true_label, sample_id = load_test_sample(args.sample_index)
        except Exception as e:
            print(f"[ERROR] Could not load sample {args.sample_index}: {e}")
            print("[INFO] Falling back to simulated sample.")
            inputs = generate_simulated_inputs(seed=args.seed)
    else:
        # Default or --simulate
        inputs = generate_simulated_inputs(seed=args.seed)
        
    # 2. Build Mask
    mask, condition_desc = get_mask(
        condition=args.condition,
        missing=args.missing,
        available=args.available,
    )
    
    # 3. Load Model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    model = create_model(checkpoint_path=ckpt_path, seed=args.seed)
    
    # 4. Perform Inference
    with torch.no_grad():
        logits, attn_weights = model(inputs, mask=mask, return_attention_weights=True)
        
    # 5. Display Results
    print_prediction_dashboard(
        inputs=inputs,
        mask=mask,
        condition_desc=condition_desc,
        logits=logits,
        attn_weights=attn_weights,
        sample_id=sample_id,
        true_label=true_label,
    )
    
    # 6. Optional JSON Save
    if args.save_json:
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).item())
        pred_code, pred_name = CANCER_CLASSES[pred_idx]
        
        result_payload = {
            "sample_id": sample_id,
            "condition": condition_desc,
            "mask": {m: float(mask[0, i].item()) for i, m in enumerate(MODALITY_KEYS)},
            "prediction": {
                "class_index": pred_idx,
                "cancer_type": pred_code,
                "full_name": pred_name,
                "confidence_pct": round(probs[pred_idx] * 100.0, 2),
            },
            "class_probabilities": {
                CANCER_CLASSES[i][0]: round(p * 100.0, 2) for i, p in enumerate(probs)
            },
        }
        if attn_weights is not None:
            result_payload["attention_weights"] = attn_weights.squeeze(0).cpu().tolist()
            
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
        print(f"[INFO] Prediction results successfully exported to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
