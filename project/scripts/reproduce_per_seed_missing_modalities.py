"""
Reproduce Per-Seed Missing Modality Evaluation (Optimized)
=========================================================
Evaluates all 10 saved checkpoints (5 seeds S1 + 5 seeds S2)
across all 15 conditions (C0 to C14) on the held-out test split (340 samples).
Exports:
1. reports/per_seed_missing_modality_results.csv (detailed per-seed metrics)
2. reports/per_seed_paired_statistical_comparison.json (paired t-tests & Wilcoxon tests)
3. artifacts/metrics/s1_s2_test_predictions.npz (all test predictions per seed & condition)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MODALITY_KEYS, EVAL_CONDITIONS
from src.models.fusion import MultimodalCrossAttentionClassifier
from src.training.metrics import compute_metrics


def run_per_seed_evaluation():
    print("=== Starting Complete Per-Seed Evaluation from Saved Checkpoints ===", flush=True)
    
    proc_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    reports_dir = PROJECT_ROOT / "reports"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    modalities = ["mRNA", "miRNA", "CNV", "Methy"]
    # Pre-convert test data to PyTorch tensors once
    test_tensors = {m: torch.tensor(np.load(proc_dir / f"{m}_test.npy"), dtype=torch.float32) for m in modalities}
    y_test = np.load(proc_dir / "labels_test.npy")
    
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "splits.csv")
    test_sids = manifest[manifest["split"] == "test"]["sample_id"].tolist()
    sample_count = len(y_test)
    assert sample_count == 340, f"Expected 340 test samples, got {sample_count}"
    
    modality_dims = {m: test_tensors[m].shape[1] for m in modalities}
    seeds = [42, 123, 456, 789, 1024]
    strategies = ["s1_complete", "s2_dropout"]
    
    rows = []
    predictions_dict = {}
    
    f1_by_condition = {c: {"s1": [], "s2": []} for c in EVAL_CONDITIONS.keys()}
    acc_by_condition = {c: {"s1": [], "s2": []} for c in EVAL_CONDITIONS.keys()}
    bacc_by_condition = {c: {"s1": [], "s2": []} for c in EVAL_CONDITIONS.keys()}
    
    for strategy in strategies:
        strat_short = "S1" if "s1" in strategy else "S2"
        for seed in seeds:
            ckpt_path = models_dir / f"fusion_{strategy}_seed_{seed}_best.pt"
            assert ckpt_path.is_file(), f"Checkpoint missing: {ckpt_path}"
            
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
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            for cond_id, cond_info in EVAL_CONDITIONS.items():
                mask_vals = [float(cond_info[k]) for k in modalities]
                mask = torch.tensor([mask_vals] * sample_count, dtype=torch.float32)
                
                with torch.no_grad():
                    logits, _ = model(test_tensors, mask)
                    preds = torch.argmax(logits, dim=1).numpy()
                    
                pred_key = f"{strat_short}_seed_{seed}_{cond_id}"
                predictions_dict[pred_key] = preds
                
                m = compute_metrics(y_test, preds)
                f1 = float(m["macro_f1"])
                acc = float(m["overall_accuracy"])
                bacc = float(m["balanced_accuracy"])
                
                if strat_short == "S1":
                    f1_by_condition[cond_id]["s1"].append(f1)
                    acc_by_condition[cond_id]["s1"].append(acc)
                    bacc_by_condition[cond_id]["s1"].append(bacc)
                else:
                    f1_by_condition[cond_id]["s2"].append(f1)
                    acc_by_condition[cond_id]["s2"].append(acc)
                    bacc_by_condition[cond_id]["s2"].append(bacc)
                    
                rows.append({
                    "strategy": strat_short,
                    "strategy_full": strategy,
                    "seed": seed,
                    "condition": cond_id,
                    "description": cond_info["description"],
                    "sample_count": sample_count,
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "macro_f1": f1,
                })
            print(f"Evaluated {strat_short} Seed {seed} across all 15 conditions.", flush=True)
                
    df = pd.DataFrame(rows)
    csv_path = reports_dir / "per_seed_missing_modality_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-seed results CSV to: {csv_path}", flush=True)
    
    np.savez_compressed(metrics_dir / "s1_s2_test_predictions.npz", **predictions_dict, y_true=y_test)
    print(f"Saved test predictions npz to: {metrics_dir / 's1_s2_test_predictions.npz'}", flush=True)
    
    # Paired statistical comparisons
    paired_stats = []
    
    for cond_id, cond_info in EVAL_CONDITIONS.items():
        s1_f1s = np.array(f1_by_condition[cond_id]["s1"])
        s2_f1s = np.array(f1_by_condition[cond_id]["s2"])
        
        diffs = s2_f1s - s1_f1s
        mean_diff = float(np.mean(diffs))
        s1_mean = float(np.mean(s1_f1s))
        s1_std = float(np.std(s1_f1s, ddof=1)) if len(s1_f1s) > 1 else 0.0
        s2_mean = float(np.mean(s2_f1s))
        s2_std = float(np.std(s2_f1s, ddof=1)) if len(s2_f1s) > 1 else 0.0
        
        diff_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        cohen_d_paired = float(mean_diff / diff_std) if diff_std > 1e-8 else 0.0
        
        if np.all(diffs == 0.0):
            t_stat, p_val_paired_t = 0.0, 1.0
            w_stat, p_val_wilcoxon = 0.0, 1.0
        else:
            t_res = stats.ttest_rel(s2_f1s, s1_f1s)
            t_stat = float(t_res.statistic)
            p_val_paired_t = float(t_res.pvalue)
            try:
                w_res = stats.wilcoxon(s2_f1s, s1_f1s, zero_method="wilcox")
                w_stat = float(w_res.statistic)
                p_val_wilcoxon = float(w_res.pvalue)
            except Exception:
                w_stat, p_val_wilcoxon = 0.0, 1.0
                
        sem_diff = diff_std / np.sqrt(len(diffs)) if diff_std > 1e-8 else 0.0
        ci95_diff_low = mean_diff - 2.776 * sem_diff
        ci95_diff_high = mean_diff + 2.776 * sem_diff
        
        paired_stats.append({
            "condition": cond_id,
            "description": cond_info["description"],
            "s1_per_seed_f1": [round(x, 6) for x in s1_f1s.tolist()],
            "s2_per_seed_f1": [round(x, 6) for x in s2_f1s.tolist()],
            "s1_mean": s1_mean,
            "s1_std": s1_std,
            "s2_mean": s2_mean,
            "s2_std": s2_std,
            "mean_gain_unrounded": mean_diff,
            "paired_diff_std": diff_std,
            "paired_cohens_d": cohen_d_paired,
            "paired_t_statistic": t_stat,
            "paired_t_pvalue": p_val_paired_t,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": p_val_wilcoxon,
            "ci95_diff_low": ci95_diff_low,
            "ci95_diff_high": ci95_diff_high,
            "statistically_significant_p05": bool(p_val_paired_t < 0.05 or p_val_wilcoxon < 0.05),
        })
        
    stats_out_path = reports_dir / "per_seed_paired_statistical_comparison.json"
    with open(stats_out_path, "w", encoding="utf-8") as f:
        json.dump(paired_stats, f, indent=2)
    print(f"Saved paired statistical analysis to: {stats_out_path}", flush=True)
    
    # Print CNV-Only (C12) detailed result
    c12_stat = next(s for s in paired_stats if s["condition"] == "C12")
    print("\n=== CNV-ONLY (C12) DEEP DIVE ===", flush=True)
    print(f"S1 Per-Seed F1: {c12_stat['s1_per_seed_f1']}", flush=True)
    print(f"S2 Per-Seed F1: {c12_stat['s2_per_seed_f1']}", flush=True)
    print(f"S1 Mean: {c12_stat['s1_mean']:.6f} +/- {c12_stat['s1_std']:.6f}", flush=True)
    print(f"S2 Mean: {c12_stat['s2_mean']:.6f} +/- {c12_stat['s2_std']:.6f}", flush=True)
    print(f"Unrounded Gain (S2 - S1): {c12_stat['mean_gain_unrounded']:.6f}", flush=True)
    print(f"Paired t-statistic: {c12_stat['paired_t_statistic']:.4f}, p-value: {c12_stat['paired_t_pvalue']:.6f}", flush=True)
    print(f"Wilcoxon statistic: {c12_stat['wilcoxon_statistic']}, p-value: {c12_stat['wilcoxon_pvalue']:.6f}", flush=True)
    print(f"Paired Cohen's d: {c12_stat['paired_cohens_d']:.4f}", flush=True)
    print(f"95% CI for Gain: [{c12_stat['ci95_diff_low']:.6f}, {c12_stat['ci95_diff_high']:.6f}]", flush=True)


if __name__ == "__main__":
    run_per_seed_evaluation()
