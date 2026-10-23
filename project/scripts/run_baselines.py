"""
Stage D: Baseline Models Execution Script
=========================================
Evaluates:
- D1: mRNA Single-Modality MLP
- D2: miRNA Single-Modality MLP
- D3: CNV Single-Modality MLP
- D4: Methy Single-Modality MLP
- D5: Concatenation (Early Fusion) MLP
- D6: LightGBM Tabular Baseline

Multi-seed evaluation across [42, 123, 456, 789, 1024].
Outputs metrics and confusion matrices to artifacts/metrics/.
"""

import sys
import json
import argparse
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
from src.models.baselines import LightGBMBaseline


def run_lightgbm_baseline(seeds: List[int]) -> Dict[str, Any]:
    """Run D6: LightGBM Tabular Baseline across seeds."""
    logger = get_logger("baseline_lightgbm")
    logger.info("Running D6: LightGBM Concatenation Baseline...")
    
    proc_dir = PROJECT_ROOT / "data" / "processed"
    
    # Load all modalities
    modalities = ["mRNA", "miRNA", "CNV", "Methy"]
    X_train_list = [np.load(proc_dir / f"{m}_train.npy") for m in modalities]
    X_val_list = [np.load(proc_dir / f"{m}_val.npy") for m in modalities]
    X_test_list = [np.load(proc_dir / f"{m}_test.npy") for m in modalities]
    
    X_train = np.concatenate(X_train_list, axis=1)
    X_val = np.concatenate(X_val_list, axis=1)
    X_test = np.concatenate(X_test_list, axis=1)
    
    y_train = np.load(proc_dir / "labels_train.npy")
    y_val = np.load(proc_dir / "labels_val.npy")
    y_test = np.load(proc_dir / "labels_test.npy")
    
    logger.info(f"Concatenated feature dimension: {X_train.shape[1]}")
    
    val_results = []
    test_results = []
    
    for seed in seeds:
        set_seed(seed)
        model = LightGBMBaseline(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        
        # Validation
        val_preds = model.predict(X_val)
        val_m = compute_metrics(y_val, val_preds)
        val_results.append(val_m)
        
        # Test
        test_preds = model.predict(X_test)
        test_m = compute_metrics(y_test, test_preds)
        test_results.append(test_m)
        
        logger.info(
            f"Seed {seed}: Val Macro-F1 = {val_m['macro_f1']:.4f}, "
            f"Test Macro-F1 = {test_m['macro_f1']:.4f}, "
            f"Test Balanced Acc = {test_m['balanced_accuracy']:.4f}"
        )
        
    val_summary = format_metric_summary(val_results)
    test_summary = format_metric_summary(test_results)
    
    logger.info(f"D6 LightGBM Validation Macro-F1: {val_summary['macro_f1']}")
    logger.info(f"D6 LightGBM Test Macro-F1: {test_summary['macro_f1']}")
    
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "experiment": "D6_LightGBM",
        "description": "Conventional tabular gradient-boosted decision tree baseline",
        "input_features": int(X_train.shape[1]),
        "seeds": seeds,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "seed_runs": [
            {"seed": s, "val": v, "test": t}
            for s, v, t in zip(seeds, val_results, test_results)
        ],
    }
    
    with open(out_dir / "baseline_d6_lightgbm.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    return result_data


def run_single_modality_mlp(modality: str, seeds: List[int]) -> Dict[str, Any]:
    """Run D1-D4 single modality MLP baseline across seeds."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.baselines import SingleModalityMLP
    from src.training.trainer import ModelTrainer
    
    logger = get_logger(f"baseline_{modality}")
    logger.info(f"Running Single-Modality MLP Baseline for {modality}...")
    
    proc_dir = PROJECT_ROOT / "data" / "processed"
    X_train = np.load(proc_dir / f"{modality}_train.npy")
    X_val = np.load(proc_dir / f"{modality}_val.npy")
    X_test = np.load(proc_dir / f"{modality}_test.npy")
    
    y_train = np.load(proc_dir / "labels_train.npy")
    y_val = np.load(proc_dir / "labels_val.npy")
    y_test = np.load(proc_dir / "labels_test.npy")
    
    sample_ids = list(range(len(y_train)))
    
    val_results = []
    test_results = []
    
    input_dim = X_train.shape[1]
    
    for seed in seeds:
        set_seed(seed)
        
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(range(len(y_train)), dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(range(len(y_val)), dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
            torch.tensor(range(len(y_test)), dtype=torch.long)
        )
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        model = SingleModalityMLP(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            num_classes=4,
            dropout=0.3,
        )
        
        trainer = ModelTrainer(
            model=model,
            learning_rate=1e-3,
            max_epochs=100,
            patience=15,
            model_name=f"mlp_{modality}_seed_{seed}",
            checkpoint_dir=PROJECT_ROOT / "artifacts" / "models",
        )
        
        trainer.fit(train_loader, val_loader)
        
        _, val_m, _, _ = trainer.evaluate(val_loader)
        _, test_m, _, _ = trainer.evaluate(test_loader)
        
        val_results.append(val_m)
        test_results.append(test_m)
        
        logger.info(
            f"[{modality}] Seed {seed}: Val Macro-F1 = {val_m['macro_f1']:.4f}, "
            f"Test Macro-F1 = {test_m['macro_f1']:.4f}"
        )
        
    val_summary = format_metric_summary(val_results)
    test_summary = format_metric_summary(test_results)
    
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "experiment": f"MLP_{modality}",
        "modality": modality,
        "input_features": input_dim,
        "seeds": seeds,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "seed_runs": [
            {"seed": s, "val": v, "test": t}
            for s, v, t in zip(seeds, val_results, test_results)
        ],
    }
    
    with open(out_dir / f"baseline_{modality.lower()}_mlp.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    return result_data


def run_early_fusion_mlp(seeds: List[int]) -> Dict[str, Any]:
    """Run D5: Concatenation (Early Fusion) MLP baseline across seeds."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import MultimodalCancerDataset
    from src.models.baselines import EarlyFusionMLP
    from src.training.trainer import ModelTrainer
    
    logger = get_logger("baseline_early_fusion")
    logger.info("Running D5: Early Fusion (Concatenation) MLP Baseline...")
    
    proc_dir = PROJECT_ROOT / "data" / "processed"
    modalities = ["mRNA", "miRNA", "CNV", "Methy"]
    
    train_data = {m: np.load(proc_dir / f"{m}_train.npy") for m in modalities}
    val_data = {m: np.load(proc_dir / f"{m}_val.npy") for m in modalities}
    test_data = {m: np.load(proc_dir / f"{m}_test.npy") for m in modalities}
    
    y_train = np.load(proc_dir / "labels_train.npy")
    y_val = np.load(proc_dir / "labels_val.npy")
    y_test = np.load(proc_dir / "labels_test.npy")
    
    with open(proc_dir / "sample_ids_full.json", "r") as f:
        all_sids = json.load(f)
        
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "splits.csv")
    train_sids = manifest[manifest["split"] == "train"]["sample_id"].tolist()
    val_sids = manifest[manifest["split"] == "val"]["sample_id"].tolist()
    test_sids = manifest[manifest["split"] == "test"]["sample_id"].tolist()
    
    modality_dims = {m: train_data[m].shape[1] for m in modalities}
    logger.info(f"Modality feature dimensions: {modality_dims}")
    
    val_results = []
    test_results = []
    
    for seed in seeds:
        set_seed(seed)
        
        train_ds = MultimodalCancerDataset(train_data, y_train, train_sids)
        val_ds = MultimodalCancerDataset(val_data, y_val, val_sids)
        test_ds = MultimodalCancerDataset(test_data, y_test, test_sids)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        model = EarlyFusionMLP(
            modality_dims=modality_dims,
            hidden_dims=[512, 256, 128],
            num_classes=4,
            dropout=0.3,
        )
        
        trainer = ModelTrainer(
            model=model,
            learning_rate=1e-3,
            max_epochs=100,
            patience=15,
            model_name=f"early_fusion_seed_{seed}",
            checkpoint_dir=PROJECT_ROOT / "artifacts" / "models",
        )
        
        trainer.fit(train_loader, val_loader)
        
        _, val_m, _, _ = trainer.evaluate(val_loader)
        _, test_m, _, _ = trainer.evaluate(test_loader)
        
        val_results.append(val_m)
        test_results.append(test_m)
        
        logger.info(
            f"[Early Fusion] Seed {seed}: Val Macro-F1 = {val_m['macro_f1']:.4f}, "
            f"Test Macro-F1 = {test_m['macro_f1']:.4f}"
        )
        
    val_summary = format_metric_summary(val_results)
    test_summary = format_metric_summary(test_results)
    
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "experiment": "D5_Early_Fusion",
        "description": "Concatenation MLP baseline",
        "total_features": sum(modality_dims.values()),
        "seeds": seeds,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "seed_runs": [
            {"seed": s, "val": v, "test": t}
            for s, v, t in zip(seeds, val_results, test_results)
        ],
    }
    
    with open(out_dir / "baseline_d5_early_fusion.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    return result_data


def main():
    parser = argparse.ArgumentParser(description="Run Stage D baseline experiments.")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "lightgbm", "mrna", "mirna", "cnv", "methy", "early_fusion"])
    args = parser.parse_args()
    
    cfg = load_config()
    seeds = cfg["experiment"]["seeds"]
    
    if args.model in ["all", "lightgbm"]:
        run_lightgbm_baseline(seeds)
        
    if args.model in ["all", "mrna"]:
        run_single_modality_mlp("mRNA", seeds)
    if args.model in ["all", "mirna"]:
        run_single_modality_mlp("miRNA", seeds)
    if args.model in ["all", "cnv"]:
        run_single_modality_mlp("CNV", seeds)
    if args.model in ["all", "methy"]:
        run_single_modality_mlp("Methy", seeds)
    if args.model in ["all", "early_fusion"]:
        run_early_fusion_mlp(seeds)


if __name__ == "__main__":
    main()
