"""
Comprehensive evaluation metrics and statistical hypothesis testing.
Adheres strictly to the evaluation protocol in docs/evaluation_protocol.md.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
)
from scipy.stats import wilcoxon


CANCER_CLASSES = ["BRCA", "HNSC", "LGG", "THCA"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CANCER_CLASSES,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics on predictions.
    
    Returns:
        Dict containing:
            macro_f1, balanced_accuracy, overall_accuracy,
            micro_f1, weighted_f1, mcc,
            per_class: {class_name: {precision, recall, f1, support}},
            confusion_matrix: 2D list
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # Per-class metrics
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    
    per_class = {}
    for i, c_name in enumerate(class_names):
        per_class[c_name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "overall_accuracy": acc,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def format_metric_summary(seed_metrics: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Given metrics across multiple seeds, computes mean ± std format for reporting.
    """
    keys = ["macro_f1", "balanced_accuracy", "overall_accuracy", "weighted_f1", "mcc"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in seed_metrics]
        mean = np.mean(vals)
        std = np.std(vals)
        summary[k] = f"{mean:.4f} ± {std:.4f}"
        summary[f"{k}_mean"] = float(mean)
        summary[f"{k}_std"] = float(std)
    return summary


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Performs Wilcoxon signed-rank test on paired seed metrics.
    """
    diff = np.array(scores_a) - np.array(scores_b)
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        stat, p = wilcoxon(scores_a, scores_b, alternative=alternative)
        return {"statistic": float(stat), "p_value": float(p)}
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "error": str(e)}
