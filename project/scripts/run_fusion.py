"""
Stage E: Multimodal Cross-Attention Fusion Model Execution
==========================================================
Trains and evaluates the proposed MultimodalCrossAttentionClassifier
under complete modality settings (Condition C0 reference) across
5 fixed random seeds: [42, 123, 456, 789, 1024].
Outputs checkpoints and detailed metric reports.
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


def run_cross_attention_fusion(
    seeds: List[int],
    training_strategy: str = "s1_complete",
    dropout_prob: float = 0.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import MultimodalCancerDataset
    from src.models.fusion import MultimodalCrossAttentionClassifier
    from src.training.trainer import ModelTrainer
    
    logger = get_logger("stage_e_fusion")
    logger.info(f"Starting Stage E Fusion Training (Strategy: {training_strategy}, dropout={dropout_prob})...")
    
    proc_dir = PROJECT_ROOT / "data" / "processed"
    modalities = ["mRNA", "miRNA", "CNV", "Methy"]
    
    train_data = {m: np.load(proc_dir / f"{m}_train.npy") for m in modalities}
    val_data = {m: np.load(proc_dir / f"{m}_val.npy") for m in modalities}
    test_data = {m: np.load(proc_dir / f"{m}_test.npy") for m in modalities}
    
    y_train = np.load(proc_dir / "labels_train.npy")
    y_val = np.load(proc_dir / "labels_val.npy")
    y_test = np.load(proc_dir / "labels_test.npy")
    
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
        logger.info(f"--- Training Fusion Model with Seed {seed} ---")
        
        train_ds = MultimodalCancerDataset(
            train_data, y_train, train_sids,
            modality_dropout_prob=dropout_prob,
            seed=seed,
        )
        val_ds = MultimodalCancerDataset(val_data, y_val, val_sids, modality_dropout_prob=0.0)
        test_ds = MultimodalCancerDataset(test_data, y_test, test_sids, modality_dropout_prob=0.0)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
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
        
        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            max_epochs=100,
            patience=15,
            model_name=f"fusion_{training_strategy}_seed_{seed}",
            checkpoint_dir=PROJECT_ROOT / "artifacts" / "models",
        )
        
        trainer.fit(train_loader, val_loader)
        
        _, val_m, _, _ = trainer.evaluate(val_loader)
        _, test_m, _, _ = trainer.evaluate(test_loader)
        
        val_results.append(val_m)
        test_results.append(test_m)
        
        logger.info(
            f"[Fusion Seed {seed}] Val Macro-F1 = {val_m['macro_f1']:.4f}, "
            f"Test Macro-F1 = {test_m['macro_f1']:.4f}, "
            f"Test Balanced Acc = {test_m['balanced_accuracy']:.4f}"
        )
        
    val_summary = format_metric_summary(val_results)
    test_summary = format_metric_summary(test_results)
    
    logger.info(f"==> Overall Val Macro-F1: {val_summary['macro_f1']}")
    logger.info(f"==> Overall Test Macro-F1: {test_summary['macro_f1']}")
    
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "experiment": f"E1_Fusion_{training_strategy}",
        "description": "Multi-head cross-attention fusion architecture",
        "modality_dims": modality_dims,
        "seeds": seeds,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "seed_runs": [
            {"seed": s, "val": v, "test": t}
            for s, v, t in zip(seeds, val_results, test_results)
        ],
    }
    
    with open(out_dir / f"fusion_{training_strategy}.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage E Cross-Attention Fusion models.")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "s1", "s2"],
                        help="Training strategy: s1 (complete only), s2 (modality dropout p=0.3), or all")
    args = parser.parse_args()
    
    cfg = load_config()
    seeds = cfg["experiment"]["seeds"]
    
    if args.strategy in ["all", "s1"]:
        run_cross_attention_fusion(seeds=seeds, training_strategy="s1_complete", dropout_prob=0.0)
    if args.strategy in ["all", "s2"]:
        run_cross_attention_fusion(seeds=seeds, training_strategy="s2_dropout", dropout_prob=0.3)

