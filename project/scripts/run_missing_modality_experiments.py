"""
Stage F: Missing-Modality Experimentation Framework
===================================================
Evaluates model robustness across all 15 conditions (C0 to C14):
- C0: Reference (all modalities present)
- C1-C4: Single modality missing
- C5-C10: Two modalities missing
- C11-C14: Three modalities missing (single modality available)

Compares Training Strategies:
- Strategy S1: Trained without modality dropout (Complete only)
- Strategy S2: Trained with modality dropout (p=0.3)

Evaluates on the test set across all 5 seeds [42, 123, 456, 789, 1024].
Outputs full degradation tables and statistical summaries.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.training.metrics import compute_metrics, format_metric_summary
from src.data.dataset import EVAL_CONDITIONS, MultimodalCancerDataset
from src.models.fusion import MultimodalCrossAttentionClassifier


def evaluate_missing_conditions(
    strategy: str = "s1_complete",
    seeds: List[int] = [42, 123, 456, 789, 1024],
    device: str = "cpu",
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    
    logger = get_logger(f"eval_missing_{strategy}")
    logger.info(f"Evaluating Strategy {strategy} under all 15 missingness conditions...")
    
    proc_dir = PROJECT_ROOT / "data" / "processed"
    modalities = ["mRNA", "miRNA", "CNV", "Methy"]
    
    test_data = {m: np.load(proc_dir / f"{m}_test.npy") for m in modalities}
    y_test = np.load(proc_dir / "labels_test.npy")
    
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "splits.csv")
    test_sids = manifest[manifest["split"] == "test"]["sample_id"].tolist()
    
    modality_dims = {m: test_data[m].shape[1] for m in modalities}
    ckpt_dir = PROJECT_ROOT / "artifacts" / "models"
    
    condition_results = {cond: [] for cond in EVAL_CONDITIONS.keys()}
    
    for seed in seeds:
        set_seed(seed)
        ckpt_path = ckpt_dir / f"fusion_{strategy}_seed_{seed}_best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train model first.")
            
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
        
        for cond_key in EVAL_CONDITIONS.keys():
            cond_ds = MultimodalCancerDataset(
                test_data, y_test, test_sids,
                fixed_condition=cond_key,
                seed=seed,
            )
            cond_loader = DataLoader(cond_ds, batch_size=64, shuffle=False)
            
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for mod_tensors, mask, labels, _ in cond_loader:
                    mod_tensors = {k: v.to(device) for k, v in mod_tensors.items()}
                    mask = mask.to(device)
                    outputs = model(mod_tensors, mask)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(labels.numpy())
                    
            m = compute_metrics(np.array(all_targets), np.array(all_preds))
            condition_results[cond_key].append(m)
            
    # Compute summary per condition
    summary_per_condition = {}
    for cond_key, metrics_list in condition_results.items():
        summary = format_metric_summary(metrics_list)
        summary["description"] = EVAL_CONDITIONS[cond_key]["description"]
        summary_per_condition[cond_key] = summary
        logger.info(
            f"[{strategy}] {cond_key} ({EVAL_CONDITIONS[cond_key]['description']}): "
            f"Macro-F1 = {summary['macro_f1']}, Balanced Acc = {summary['balanced_accuracy']}"
        )
        
    return {
        "strategy": strategy,
        "seeds": seeds,
        "conditions": summary_per_condition,
        "raw_results": condition_results,
    }


def compare_and_report_strategies(
    s1_results: Dict[str, Any],
    s2_results: Dict[str, Any],
) -> pd.DataFrame:
    """Compare S1 vs S2 across all conditions and export CSV and JSON reports."""
    rows = []
    
    for cond_key in EVAL_CONDITIONS.keys():
        desc = EVAL_CONDITIONS[cond_key]["description"]
        s1_f1 = s1_results["conditions"][cond_key]["macro_f1"]
        s2_f1 = s2_results["conditions"][cond_key]["macro_f1"]
        s1_mean = s1_results["conditions"][cond_key]["macro_f1_mean"]
        s2_mean = s2_results["conditions"][cond_key]["macro_f1_mean"]
        gain = s2_mean - s1_mean
        
        rows.append({
            "Condition": cond_key,
            "Description": desc,
            "S1_NoDropout_F1": s1_f1,
            "S2_ModalityDropout_F1": s2_f1,
            "S1_Mean": round(s1_mean, 4),
            "S2_Mean": round(s2_mean, 4),
            "Dropout_Gain": round(gain, 4),
        })
        
    df = pd.DataFrame(rows)
    
    reports_dir = PROJECT_ROOT / "reports"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = reports_dir / "missing_modality_summary.csv"
    df.to_csv(csv_path, index=False)
    
    with open(metrics_dir / "missing_modality_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "s1_no_dropout": s1_results["conditions"],
            "s2_modality_dropout": s2_results["conditions"],
            "comparison_table": rows,
        }, f, indent=2)
        
    return df


def main():
    cfg = load_config()
    seeds = cfg["experiment"]["seeds"]
    
    logger = get_logger("run_missing_experiments")
    logger.info("Executing Stage F Missing Modality Robustness Evaluation...")
    
    # 1. Evaluate Strategy S1
    s1_results = evaluate_missing_conditions(strategy="s1_complete", seeds=seeds)
    
    # 2. Evaluate Strategy S2
    s2_results = evaluate_missing_conditions(strategy="s2_dropout", seeds=seeds)
    
    # 3. Compare and Report
    df = compare_and_report_strategies(s1_results, s2_results)
    logger.info(f"\nMissing Modality Robustness Comparison:\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
