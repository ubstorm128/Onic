"""
Stage G: Explainability and Biomarker Discovery Script
======================================================
Executes:
1. Modality-Level Importance (Ablation, Permutation, Cross-Attention)
2. Feature-Level Importance (Input * Gradient Attribution)
3. Biological Marker Extraction for BRCA, HNSC, LGG, THCA
4. Generates comprehensive markdown report and metrics JSON
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.dataset import MODALITY_KEYS, MultimodalCancerDataset
from src.models.fusion import MultimodalCrossAttentionClassifier
from src.explainability.modality_importance import (
    compute_modality_ablation_importance,
    compute_modality_permutation_importance,
    analyze_attention_weights,
)
from src.explainability.feature_importance import (
    compute_feature_attributions,
    get_top_features_per_class,
)


def run_explainability_pipeline(
    model_strategy: str = "s2_dropout",
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    logger = get_logger("stage_g_explainability")
    logger.info("Starting Stage G Explainability Pipeline...")
    
    set_seed(seed)
    proc_dir = PROJECT_ROOT / "data" / "processed"
    
    # Load processed test split
    test_data = {m: np.load(proc_dir / f"{m}_test.npy") for m in MODALITY_KEYS}
    y_test = np.load(proc_dir / "labels_test.npy")
    
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "splits.csv")
    test_sids = manifest[manifest["split"] == "test"]["sample_id"].tolist()
    
    modality_dims = {m: test_data[m].shape[1] for m in MODALITY_KEYS}
    
    # Load feature names from metadata
    with open(proc_dir / "metadata.json", "r") as f:
        meta = json.load(f)
        
    feature_names = {}
    for m in MODALITY_KEYS:
        # Load fitted preprocessor to get exact selected feature names
        prep_path = proc_dir / f"preprocessor_{m}.joblib"
        import joblib
        prep = joblib.load(prep_path)
        feature_names[m] = prep.selected_feature_names
        
    # Load trained model
    ckpt_path = PROJECT_ROOT / "artifacts" / "models" / f"fusion_{model_strategy}_seed_{seed}_best.pt"
    if not ckpt_path.is_file():
        # Fallback to s1 if s2 not present yet
        ckpt_path = PROJECT_ROOT / "artifacts" / "models" / f"fusion_s1_complete_seed_{seed}_best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
            
    logger.info(f"Loading trained model checkpoint: {ckpt_path.name}")
    model = MultimodalCrossAttentionClassifier(
        modality_dims=modality_dims,
        encoder_hidden_dim=256,
        latent_dim=128,
        num_heads=4,
        num_layers=1,
        dim_feedforward=256,
        num_classes=4,
        use_modality_type_embeddings=True,
        use_learned_default_token=True,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    test_ds = MultimodalCancerDataset(test_data, y_test, test_sids, modality_dropout_prob=0.0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 1. Modality Ablation Importance
    logger.info("Computing modality ablation importance...")
    ablation_imp = compute_modality_ablation_importance(model, test_loader, device=device)
    logger.info(f"Ablation importance: {ablation_imp}")
    
    # 2. Modality Permutation Importance
    logger.info("Computing modality permutation importance (5 repeats)...")
    perm_imp = compute_modality_permutation_importance(
        model, test_data, y_test, test_sids, device=device, n_repeats=5, seed=seed
    )
    logger.info(f"Permutation importance: {perm_imp}")
    
    # 3. Inter-Modality Attention Analysis
    logger.info("Analyzing inter-modality cross-attention weights...")
    mean_attn = analyze_attention_weights(model, test_loader, device=device)
    logger.info(f"Mean attention weights matrix:\n{mean_attn}")
    
    # 4. Feature-level Attributions per Class
    logger.info("Computing Input * Gradient feature attributions per cancer type...")
    class_names = ["BRCA", "HNSC", "LGG", "THCA"]
    attributions_per_class = {}
    
    for c_idx in range(4):
        attributions_per_class[c_idx] = compute_feature_attributions(
            model=model,
            X_dict=test_data,
            y=y_test,
            target_class=c_idx,
            device=device,
        )
        
    top_features = get_top_features_per_class(
        attributions_per_class=attributions_per_class,
        feature_names_dict=feature_names,
        top_k=10,
        class_names=class_names,
    )
    
    # Export reports
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    reports_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    explainability_results = {
        "model_strategy": model_strategy,
        "seed": seed,
        "modality_ablation": ablation_imp,
        "modality_permutation": perm_imp,
        "inter_modality_attention": {
            "modalities": MODALITY_KEYS,
            "matrix": mean_attn.tolist(),
        },
        "top_features_per_cancer": top_features,
    }
    
    with open(out_dir / "explainability_report.json", "w", encoding="utf-8") as f:
        json.dump(explainability_results, f, indent=2)
        
    # Write summary markdown report
    md_lines = [
        "# Stage G: Explainability and Biomarker Discovery Report",
        "",
        f"**Model Configuration:** {model_strategy} (Seed {seed})  ",
        "**Cohort:** TCGA Pan-Cancer (BRCA, HNSC, LGG, THCA; N=2,263, Test N=340)",
        "",
        "---",
        "",
        "## 1. Modality-Level Importance",
        "",
        "### 1.1 Ablation and Permutation Drop",
        "",
        "| Modality | Ablated F1 | Drop from C0 (Ablation) | Permutation Mean Drop ± Std |",
        "|---|---:|---:|---:|",
    ]
    
    for mod in MODALITY_KEYS:
        abl_f1 = ablation_imp.get(f"{mod}_ablated_f1", 0.0)
        abl_drop = ablation_imp.get(f"{mod}_importance_drop", 0.0)
        p_drop = perm_imp[mod]["mean_drop"]
        p_std = perm_imp[mod]["std_drop"]
        md_lines.append(f"| **{mod}** | {abl_f1:.4f} | {abl_drop:+.4f} | {p_drop:.4f} ± {p_std:.4f} |")
        
    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Inter-Modality Attention Weights",
        "",
        "> **Methodological Limitation Notice:** Self-attention weights reflect token interactions in the fusion layer, not causal attribution. They are reported to characterize information routing across omics tokens.",
        "",
        f"| Query \\ Key | {' | '.join(MODALITY_KEYS)} |",
        f"|---|{'---:|' * len(MODALITY_KEYS)}",
    ])
    
    for i, q_mod in enumerate(MODALITY_KEYS):
        row_vals = " | ".join(f"{mean_attn[i, j]:.4f}" for j in range(len(MODALITY_KEYS)))
        md_lines.append(f"| **{q_mod}** | {row_vals} |")
        
    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Top Biological Biomarkers per Cancer Type (mRNA)",
        "",
    ])
    
    for c_name in class_names:
        md_lines.append(f"### {c_name}")
        md_lines.append("| Rank | Gene (mRNA) | Attribution Score | Top miRNA | Top CNV | Top Methylation Probe |")
        md_lines.append("|---:|---|---:|---|---|---|")
        for r in range(min(5, len(top_features[c_name]["mRNA"]))):
            g_item = top_features[c_name]["mRNA"][r]
            mir_item = top_features[c_name]["miRNA"][r]
            cnv_item = top_features[c_name]["CNV"][r]
            meth_item = top_features[c_name]["Methy"][r]
            md_lines.append(
                f"| {r+1} | `{g_item['feature']}` | {g_item['attribution_score']:.6f} | "
                f"`{mir_item['feature']}` | `{cnv_item['feature']}` | `{meth_item['feature']}` |"
            )
        md_lines.append("")
        
    report_path = reports_dir / "explainability_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    logger.info(f"Stage G successfully completed! Reports saved to {report_path} and {out_dir}")
    return explainability_results


if __name__ == "__main__":
    run_explainability_pipeline()
