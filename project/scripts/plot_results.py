"""
Visualization and Publication Figures Generation Script
========================================================
Generates publication-ready figures:
1. Fig 1: Baseline Models Comparison (Macro-F1 and Balanced Accuracy)
2. Fig 2: Missing Modality Robustness Curves (Strategy S1 vs Strategy S2 across C0-C14)
3. Fig 3: Inter-Modality Cross-Attention Heatmap (4x4 Matrix)
4. Fig 4: Top Biological Biomarkers per Cancer Type (mRNA Gene Attributions)
5. Fig 5: Confusion Matrix for Proposed Model under Complete & Missing Conditions
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_plot_style():
    """Apply modern publication aesthetics."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14


def plot_baselines_comparison(out_dir: Path):
    """Figure 1: Baseline comparison bar chart."""
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    
    models = ["mRNA (MLP)", "miRNA (MLP)", "CNV (MLP)", "Methy (MLP)", "Early Fusion (MLP)", "LightGBM (GBDT)", "Proposed Fusion"]
    files = [
        metrics_dir / "baseline_mrna_mlp.json",
        metrics_dir / "baseline_mirna_mlp.json",
        metrics_dir / "baseline_cnv_mlp.json",
        metrics_dir / "baseline_methy_mlp.json",
        metrics_dir / "baseline_d5_early_fusion.json",
        metrics_dir / "baseline_d6_lightgbm.json",
        metrics_dir / "fusion_s1_complete.json",
    ]
    
    f1_means, f1_stds = [], []
    for f in files:
        if f.is_file():
            with open(f, "r") as fp:
                d = json.load(fp)
            f1_means.append(d["test_summary"]["macro_f1_mean"])
            f1_stds.append(d["test_summary"]["macro_f1_std"])
        else:
            f1_means.append(0.0)
            f1_stds.append(0.0)
            
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#E24A33"]
    bars = ax.bar(models, f1_means, yerr=f1_stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    
    ax.set_ylabel("Test Macro-F1 Score")
    ax.set_title("Multi-Cancer Classification Performance Across Baseline & Proposed Architectures")
    ax.set_ylim(0.7, 1.02)
    plt.xticks(rotation=25, ha="right")
    
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
            
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_baselines_comparison.png")
    plt.close()


def plot_missing_modality_curves(out_dir: Path):
    """Figure 2: Robustness curves across missing conditions (S1 vs S2)."""
    comp_file = PROJECT_ROOT / "artifacts" / "metrics" / "missing_modality_comparison.json"
    if not comp_file.is_file():
        return
        
    with open(comp_file, "r") as f:
        data = json.load(f)
        
    table = data["comparison_table"]
    conditions = [r["Condition"] for r in table]
    s1_scores = [r["S1_Mean"] for r in table]
    s2_scores = [r["S2_Mean"] for r in table]
    
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    ax.plot(conditions, s1_scores, marker="o", color="#C44E52", linewidth=2.2, label="Strategy S1: No Dropout (Standard)")
    ax.plot(conditions, s2_scores, marker="s", color="#2B7A78", linewidth=2.2, label="Strategy S2: Modality Dropout (Robust)")
    
    ax.fill_between(conditions, s1_scores, s2_scores, where=np.array(s2_scores) >= np.array(s1_scores),
                    color="#2B7A78", alpha=0.15, label="Dropout Robustness Gain")
    
    ax.set_xlabel("Missing Modality Condition (C0: Complete, C1-C4: 1 Missing, C5-C10: 2 Missing, C11-C14: 3 Missing)")
    ax.set_ylabel("Test Macro-F1 Score")
    ax.set_title("Robustness Under Simulated Missing Modalities: Strategy S1 vs S2")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_missing_modality_robustness.png")
    plt.close()


def plot_attention_heatmap(out_dir: Path):
    """Figure 3: Inter-modality attention heatmap."""
    exp_file = PROJECT_ROOT / "artifacts" / "metrics" / "explainability_report.json"
    if not exp_file.is_file():
        return
        
    with open(exp_file, "r") as f:
        data = json.load(f)
        
    matrix = np.array(data["inter_modality_attention"]["matrix"])
    modalities = data["inter_modality_attention"]["modalities"]
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=modalities, yticklabels=modalities,
                cbar_kws={"label": "Attention Weight"}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel("Key Token (Attended Modality)")
    ax.set_ylabel("Query Token (Origin Modality)")
    ax.set_title("Cross-Attention Information Routing Across Omics Modalities")
    
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_attention_heatmap.png")
    plt.close()


def plot_ablation_study(out_dir: Path):
    """Figure 4: Ablation study showing impact of hyperparameters on C0 and C12 performance."""
    abl_file = PROJECT_ROOT / "artifacts" / "metrics" / "ablation_study.json"
    if not abl_file.is_file():
        return
        
    with open(abl_file, "r") as f:
        data = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    
    # 1. Masking Mechanism
    ax1 = axes[0, 0]
    labels1 = ["Learned Default Token", "Zero Masking"]
    c0_1 = [data["default_vs_zero_learned"]["c0_macro_f1_mean"], data["default_vs_zero_zeros"]["c0_macro_f1_mean"]]
    c12_1 = [data["default_vs_zero_learned"]["c12_macro_f1_mean"], data["default_vs_zero_zeros"]["c12_macro_f1_mean"]]
    c12_err_1 = [data["default_vs_zero_learned"]["c12_macro_f1_std"], data["default_vs_zero_zeros"]["c12_macro_f1_std"]]
    
    x = np.arange(len(labels1))
    w = 0.35
    ax1.bar(x - w/2, c0_1, w, label="Complete (C0)", color="#2b5c8f")
    ax1.bar(x + w/2, c12_1, w, yerr=c12_err_1, capsize=4, label="Only CNV (C12)", color="#d95f02")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels1)
    ax1.set_ylabel("Macro-F1 Score")
    ax1.set_title("A. Missing Modality Handling")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc="lower right")
    
    # 2. Modality Dropout Rate
    ax2 = axes[0, 1]
    p_vals = ["p = 0.1", "p = 0.3", "p = 0.5"]
    c0_p = [data["dropout_p_0.1"]["c0_macro_f1_mean"], data["default_vs_zero_learned"]["c0_macro_f1_mean"], data["dropout_p_0.5"]["c0_macro_f1_mean"]]
    c12_p = [data["dropout_p_0.1"]["c12_macro_f1_mean"], data["default_vs_zero_learned"]["c12_macro_f1_mean"], data["dropout_p_0.5"]["c12_macro_f1_mean"]]
    c12_err_p = [data["dropout_p_0.1"]["c12_macro_f1_std"], data["default_vs_zero_learned"]["c12_macro_f1_std"], data["dropout_p_0.5"]["c12_macro_f1_std"]]
    
    xp = np.arange(len(p_vals))
    ax2.plot(xp, c0_p, marker="o", lw=2, color="#2b5c8f", label="Complete (C0)")
    ax2.errorbar(xp, c12_p, yerr=c12_err_p, marker="s", lw=2, capsize=4, color="#d95f02", label="Only CNV (C12)")
    ax2.set_xticks(xp)
    ax2.set_xticklabels(p_vals)
    ax2.set_ylabel("Macro-F1 Score")
    ax2.set_title("B. Training Modality Dropout Rate (p)")
    ax2.set_ylim(0.55, 1.05)
    ax2.legend(loc="center right")
    
    # 3. Latent Representation Dimension
    ax3 = axes[1, 0]
    dims = ["D = 64", "D = 128", "D = 256"]
    c0_d = [data["latent_dim_64"]["c0_macro_f1_mean"], data["default_vs_zero_learned"]["c0_macro_f1_mean"], data["latent_dim_256"]["c0_macro_f1_mean"]]
    c12_d = [data["latent_dim_64"]["c12_macro_f1_mean"], data["default_vs_zero_learned"]["c12_macro_f1_mean"], data["latent_dim_256"]["c12_macro_f1_mean"]]
    c12_err_d = [data["latent_dim_64"]["c12_macro_f1_std"], data["default_vs_zero_learned"]["c12_macro_f1_std"], data["latent_dim_256"]["c12_macro_f1_std"]]
    
    xd = np.arange(len(dims))
    ax3.plot(xd, c0_d, marker="o", lw=2, color="#2b5c8f", label="Complete (C0)")
    ax3.errorbar(xd, c12_d, yerr=c12_err_d, marker="s", lw=2, capsize=4, color="#d95f02", label="Only CNV (C12)")
    ax3.set_xticks(xd)
    ax3.set_xticklabels(dims)
    ax3.set_ylabel("Macro-F1 Score")
    ax3.set_title("C. Latent Representation Dimension (D)")
    ax3.set_ylim(0.55, 1.05)
    ax3.legend(loc="lower right")
    
    # 4. Multi-Head Attention Heads
    ax4 = axes[1, 1]
    heads = ["H = 1 (Single-Head)", "H = 2", "H = 4"]
    c0_h = [data["heads_1"]["c0_macro_f1_mean"], data["heads_2"]["c0_macro_f1_mean"], data["default_vs_zero_learned"]["c0_macro_f1_mean"]]
    c12_h = [data["heads_1"]["c12_macro_f1_mean"], data["heads_2"]["c12_macro_f1_mean"], data["default_vs_zero_learned"]["c12_macro_f1_mean"]]
    c12_err_h = [data["heads_1"]["c12_macro_f1_std"], data["heads_2"]["c12_macro_f1_std"], data["default_vs_zero_learned"]["c12_macro_f1_std"]]
    
    xh = np.arange(len(heads))
    ax4.plot(xh, c0_h, marker="o", lw=2, color="#2b5c8f", label="Complete (C0)")
    ax4.errorbar(xh, c12_h, yerr=c12_err_h, marker="s", lw=2, capsize=4, color="#d95f02", label="Only CNV (C12)")
    ax4.set_xticks(xh)
    ax4.set_xticklabels(heads)
    ax4.set_ylabel("Macro-F1 Score")
    ax4.set_title("D. Attention Head Count (H)")
    ax4.set_ylim(0.45, 1.05)
    ax4.legend(loc="lower right")
    
    plt.suptitle("Figure 4: Hyperparameter Ablation Study & Missing Modality Stress Response", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_ablation_study.png", bbox_inches="tight")
    plt.close()


def generate_all_figures():
    set_plot_style()
    out_dir = PROJECT_ROOT / "artifacts" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_baselines_comparison(out_dir)
    plot_missing_modality_curves(out_dir)
    plot_attention_heatmap(out_dir)
    plot_ablation_study(out_dir)
    print(f"[PLOTS] Publication figures saved to {out_dir}")


if __name__ == "__main__":
    generate_all_figures()
