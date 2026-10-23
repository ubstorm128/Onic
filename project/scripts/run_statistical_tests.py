"""
Statistical Hypothesis Testing & Significance Analysis (Corrected & Paired)
==========================================================================
1. Evaluates S1 vs S2 using exact per-seed pairs on the same test set across all 15 conditions.
2. Computes paired mean difference (D_bar), paired standard deviation (s_D), paired Cohen's d,
   paired Student's t-test (df=4), Wilcoxon signed-rank test, and 95% Confidence Intervals.
3. Computes baseline confidence intervals strictly from actual seed_runs (never fabricating observations).
4. Clearly documents that 5 seeds represent training initialization/stochasticity variance,
   not independent patient sample variance.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger
from src.data.dataset import EVAL_CONDITIONS


def run_statistical_analysis():
    logger = get_logger("statistical_tests")
    logger.info("Starting Rigorous Paired Statistical Significance Analysis...")
    
    reports_dir = PROJECT_ROOT / "reports"
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    
    per_seed_csv = reports_dir / "per_seed_missing_modality_results.csv"
    if not per_seed_csv.is_file():
        raise FileNotFoundError(f"Missing {per_seed_csv}. Run reproduce_per_seed_missing_modalities.py first.")
        
    df_seeds = pd.read_csv(per_seed_csv)
    
    # 1. Paired Comparison between S1 and S2 across conditions
    paired_results = []
    
    for cond_id, cond_info in EVAL_CONDITIONS.items():
        s1_sub = df_seeds[(df_seeds["strategy"] == "S1") & (df_seeds["condition"] == cond_id)].sort_values("seed")
        s2_sub = df_seeds[(df_seeds["strategy"] == "S2") & (df_seeds["condition"] == cond_id)].sort_values("seed")
        
        seeds_s1 = s1_sub["seed"].tolist()
        seeds_s2 = s2_sub["seed"].tolist()
        assert seeds_s1 == seeds_s2, f"Seed mismatch in {cond_id}: {seeds_s1} vs {seeds_s2}"
        
        s1_f1 = s1_sub["macro_f1"].to_numpy(dtype=np.float64)
        s2_f1 = s2_sub["macro_f1"].to_numpy(dtype=np.float64)
        
        diffs = s2_f1 - s1_f1
        n = len(diffs)
        assert n == 5, f"Expected 5 seeds, got {n}"
        
        mean_diff = float(np.mean(diffs))
        s1_mean = float(np.mean(s1_f1))
        s2_mean = float(np.mean(s2_f1))
        
        s1_std = float(np.std(s1_f1, ddof=1))
        s2_std = float(np.std(s2_f1, ddof=1))
        diff_std = float(np.std(diffs, ddof=1))
        
        # Paired Cohen's d: mean(diff) / std(diff)
        if diff_std > 1e-9:
            paired_cohen_d = float(mean_diff / diff_std)
        else:
            paired_cohen_d = 0.0
            
        # Paired t-test
        if np.all(diffs == 0.0) or diff_std < 1e-9:
            t_stat = 0.0
            p_val_t = 1.0
            w_stat = 0.0
            p_val_w = 1.0
        else:
            t_res = stats.ttest_rel(s2_f1, s1_f1)
            t_stat = float(t_res.statistic)
            p_val_t = float(t_res.pvalue)
            
            try:
                # Wilcoxon signed-rank test
                w_res = stats.wilcoxon(s2_f1, s1_f1, zero_method="wilcox")
                w_stat = float(w_res.statistic)
                p_val_w = float(w_res.pvalue)
            except Exception:
                w_stat = 0.0
                p_val_w = 1.0
                
        # 95% Confidence interval for paired difference: t(4, 0.975) = 2.7764451
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        sem_diff = float(diff_std / np.sqrt(n)) if diff_std > 1e-9 else 0.0
        ci95_low = float(mean_diff - t_crit * sem_diff)
        ci95_high = float(mean_diff + t_crit * sem_diff)
        
        paired_results.append({
            "condition": cond_id,
            "description": cond_info["description"],
            "num_seeds": n,
            "seeds": seeds_s1,
            "s1_per_seed_f1": s1_f1.tolist(),
            "s2_per_seed_f1": s2_f1.tolist(),
            "s1_mean": s1_mean,
            "s1_std": s1_std,
            "s2_mean": s2_mean,
            "s2_std": s2_std,
            "paired_mean_difference": mean_diff,
            "paired_difference_std": diff_std,
            "paired_difference_sem": sem_diff,
            "paired_cohens_d": paired_cohen_d,
            "paired_t_statistic": t_stat,
            "paired_t_pvalue": p_val_t,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": p_val_w,
            "ci95_difference_low": ci95_low,
            "ci95_difference_high": ci95_high,
            "is_significant_alpha_05": bool(p_val_t < 0.05),
        })
        
    # 2. Baseline Model Confidence Intervals (strictly from actual seed_runs)
    baseline_files = {
        "D1_mRNA_MLP": "baseline_mrna_mlp.json",
        "D2_miRNA_MLP": "baseline_mirna_mlp.json",
        "D3_CNV_MLP": "baseline_cnv_mlp.json",
        "D4_Methy_MLP": "baseline_methy_mlp.json",
        "D5_Early_Fusion_MLP": "baseline_d5_early_fusion.json",
        "D6_LightGBM": "baseline_d6_lightgbm.json",
        "E1_Proposed_S1": "fusion_s1_complete.json",
        "E2_Proposed_S2": "fusion_s2_dropout.json",
    }
    
    baseline_summaries = {}
    for name, fname in baseline_files.items():
        fpath = metrics_dir / fname
        if not fpath.is_file():
            logger.warning(f"File not found: {fpath}")
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
            
        # Extract per-seed test macro-F1 strictly from actual seed_runs
        if "seed_runs" in d and len(d["seed_runs"]) > 0:
            test_f1s = [float(run["test"]["macro_f1"]) for run in d["seed_runs"]]
            n_seeds = len(test_f1s)
            mean_f1 = float(np.mean(test_f1s))
            std_f1 = float(np.std(test_f1s, ddof=1)) if n_seeds > 1 else 0.0
            sem_f1 = float(std_f1 / np.sqrt(n_seeds)) if n_seeds > 1 and std_f1 > 1e-9 else 0.0
            
            if n_seeds > 1 and sem_f1 > 0.0:
                t_crit = float(stats.t.ppf(0.975, df=n_seeds - 1))
                ci_low = float(mean_f1 - t_crit * sem_f1)
                ci_high = float(mean_f1 + t_crit * sem_f1)
            else:
                ci_low = mean_f1
                ci_high = mean_f1
                
            baseline_summaries[name] = {
                "num_seeds": n_seeds,
                "per_seed_f1": test_f1s,
                "mean_macro_f1": mean_f1,
                "sample_std": std_f1,
                "sem": sem_f1,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "has_seed_variance": bool(std_f1 > 1e-9),
            }
        else:
            logger.error(f"Cannot extract individual seed runs for {name}. No fabrication allowed.")
            
    # Export Machine-Readable JSON (full precision)
    full_output = {
        "metadata": {
            "description": "Statistically rigorous paired comparisons (S1 vs S2) and baseline confidence intervals",
            "eval_sample_count": 340,
            "training_seeds": [42, 123, 456, 789, 1024],
            "scope_qualification": "Confidence intervals and paired tests quantify algorithmic training variance (initialization and dropout stochasticity) on a fixed test split of 340 patients, not independent patient population resample variance.",
        },
        "paired_missing_modality_comparisons": paired_results,
        "baseline_model_performance": baseline_summaries,
    }
    
    json_path = metrics_dir / "statistical_significance_tests.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved full-precision statistical JSON to: {json_path}")
    
    # Export CSV summary
    csv_rows = []
    for r in paired_results:
        csv_rows.append({
            "Condition": r["condition"],
            "Description": r["description"],
            "S1_Mean": r["s1_mean"],
            "S1_Std": r["s1_std"],
            "S2_Mean": r["s2_mean"],
            "S2_Std": r["s2_std"],
            "Paired_Mean_Diff": r["paired_mean_difference"],
            "Paired_Diff_Std": r["paired_difference_std"],
            "Paired_Cohens_d": r["paired_cohens_d"],
            "Paired_t_stat": r["paired_t_statistic"],
            "Paired_t_pval": r["paired_t_pvalue"],
            "Wilcoxon_stat": r["wilcoxon_statistic"],
            "Wilcoxon_pval": r["wilcoxon_pvalue"],
            "CI95_Diff_Low": r["ci95_difference_low"],
            "CI95_Diff_High": r["ci95_difference_high"],
            "Significant_p05": r["is_significant_alpha_05"],
        })
    df_stat_csv = pd.DataFrame(csv_rows)
    csv_path = reports_dir / "statistical_significance_report.csv"
    df_stat_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved statistical CSV to: {csv_path}")
    
    # Export Markdown Report
    md_lines = [
        "# Statistical Significance and Paired Hypothesis Testing Report",
        "",
        "**Methodological Framework:**",
        "- **Paired Comparison:** S1 (complete reference) and S2 (modality dropout) are paired seed-by-seed ($N=5$) on the held-out test split of 340 patients.",
        "- **Variance Interpretation:** All statistics quantify **algorithmic training variance** (initialization, mini-batch ordering, and stochastic dropout masks). They do **not** represent patient population sampling variance, as all models evaluated the same 340 test patients.",
        "- **Wilcoxon Limitation:** With $N=5$ pairs, the minimum possible two-sided Wilcoxon $p$-value is $2/32 = 0.0625$ (when all 5 pairs show positive difference and zero ties). Hence, Wilcoxon signed-rank tests cannot reach $\\alpha = 0.05$ with $N=5$.",
        "",
        "## 1. Paired Missing-Modality Evaluation (S2 vs. S1 Across 15 Conditions)",
        "",
        "| Condition | Description | S1 Mean ± Std | S2 Mean ± Std | Paired Mean Diff (Δ) | Paired Cohen's d | Paired t-stat | Paired t p-val | Wilcoxon stat (p-val) | 95% CI of Difference | Significant (α=0.05) |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in paired_results:
        sig_str = "Yes" if r["is_significant_alpha_05"] else "No"
        md_lines.append(
            f"| **{r['condition']}** | {r['description']} | "
            f"{r['s1_mean']:.4f} ± {r['s1_std']:.4f} | {r['s2_mean']:.4f} ± {r['s2_std']:.4f} | "
            f"**{r['paired_mean_difference']:+.6f}** | {r['paired_cohens_d']:.4f} | "
            f"{r['paired_t_statistic']:.4f} | {r['paired_t_pvalue']:.4f} | "
            f"{r['wilcoxon_statistic']:.1f} ({r['wilcoxon_pvalue']:.4f}) | "
            f"[{r['ci95_difference_low']:+.4f}, {r['ci95_difference_high']:+.4f}] | {sig_str} |"
        )
        
    md_lines.extend([
        "",
        "## 2. Baseline Model Performance & 95% Confidence Intervals",
        "",
        "*(Computed strictly from individual seed test runs. Zero fabrication).* ",
        "",
        "| Model | Modality Scope | N Seeds | Raw Per-Seed Test Macro-F1 | Mean ± Sample Std | 95% Confidence Interval | Seed Variance Detected? |",
        "|:---|:---:|:---:|:---|:---:|:---:|:---:|",
    ])
    for name, b in baseline_summaries.items():
        var_str = "Yes" if b["has_seed_variance"] else "No (Deterministic/Zero)"
        raw_seeds_str = "[" + ", ".join(f"{x:.4f}" for x in b["per_seed_f1"]) + "]"
        md_lines.append(
            f"| **{name}** | Complete / Single | {b['num_seeds']} | {raw_seeds_str} | "
            f"{b['mean_macro_f1']:.4f} ± {b['sample_std']:.4f} | "
            f"[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}] | {var_str} |"
        )
        
    md_report_path = reports_dir / "statistical_significance_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved Markdown statistical report to: {md_report_path}")
    print("=== Statistical Analysis Successfully Regenerated ===")


if __name__ == "__main__":
    run_statistical_analysis()
