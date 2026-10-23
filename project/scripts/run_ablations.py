"""
Ablation Studies Script
=======================
Systematically evaluates:
1. Learned Default Token vs. Zero Masking (Ablation A)
2. Modality Dropout Rate p in {0.1, 0.3, 0.5} (Ablation B)
3. Latent Representation Dimension D in {64, 128, 256} (Ablation C)
4. Multi-Head Attention Heads H in {1, 2, 4} (Ablation D)

Evaluates each configuration on Test C0 (complete) and Test C12 (CNV only, the hardest stress test)
across fixed random seeds.
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

from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.dataset import MultimodalCancerDataset, EVAL_CONDITIONS
from src.models.fusion import MultimodalCrossAttentionClassifier
from src.training.trainer import ModelTrainer
from src.training.metrics import compute_metrics, format_metric_summary


def run_ablation_experiments(seeds: List[int] = [42, 123, 456]):
    logger = get_logger("ablation_study")
    logger.info("Starting Systematic Ablation Experiments...")
    
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
    
    # Define ablation matrix
    # Default baseline configuration:
    # latent_dim=128, num_heads=4, dropout=0.3, use_learned_default_token=True
    experiments = [
        # Group 1: Default Token vs Zero Masking
        {
            "name": "default_vs_zero_learned",
            "group": "masking_mechanism",
            "params": {"latent_dim": 128, "num_heads": 4, "dropout_prob": 0.3, "use_learned_default": True},
        },
        {
            "name": "default_vs_zero_zeros",
            "group": "masking_mechanism",
            "params": {"latent_dim": 128, "num_heads": 4, "dropout_prob": 0.3, "use_learned_default": False},
        },
        # Group 2: Modality Dropout Probability
        {
            "name": "dropout_p_0.1",
            "group": "dropout_rate",
            "params": {"latent_dim": 128, "num_heads": 4, "dropout_prob": 0.1, "use_learned_default": True},
        },
        {
            "name": "dropout_p_0.5",
            "group": "dropout_rate",
            "params": {"latent_dim": 128, "num_heads": 4, "dropout_prob": 0.5, "use_learned_default": True},
        },
        # Group 3: Latent Dimension
        {
            "name": "latent_dim_64",
            "group": "latent_dimension",
            "params": {"latent_dim": 64, "num_heads": 4, "dropout_prob": 0.3, "use_learned_default": True},
        },
        {
            "name": "latent_dim_256",
            "group": "latent_dimension",
            "params": {"latent_dim": 256, "num_heads": 4, "dropout_prob": 0.3, "use_learned_default": True},
        },
        # Group 4: Attention Heads
        {
            "name": "heads_1",
            "group": "attention_heads",
            "params": {"latent_dim": 128, "num_heads": 1, "dropout_prob": 0.3, "use_learned_default": True},
        },
        {
            "name": "heads_2",
            "group": "attention_heads",
            "params": {"latent_dim": 128, "num_heads": 2, "dropout_prob": 0.3, "use_learned_default": True},
        },
    ]
    
    results = {}
    
    for exp in experiments:
        exp_name = exp["name"]
        params = exp["params"]
        logger.info(f"Running Ablation: {exp_name} ({params})")
        
        c0_scores = []
        c12_scores = []
        
        for seed in seeds:
            set_seed(seed)
            train_ds = MultimodalCancerDataset(
                train_data, y_train, train_sids,
                modality_dropout_prob=params["dropout_prob"],
                seed=seed,
            )
            val_ds = MultimodalCancerDataset(val_data, y_val, val_sids, modality_dropout_prob=0.0)
            
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
            
            model = MultimodalCrossAttentionClassifier(
                modality_dims=modality_dims,
                encoder_hidden_dim=256,
                latent_dim=params["latent_dim"],
                num_heads=params["num_heads"],
                num_layers=1,
                dim_feedforward=256,
                num_classes=4,
                encoder_dropout=0.3,
                fusion_dropout=0.1,
                classifier_dropout=0.3,
                use_modality_type_embeddings=True,
                use_learned_default_token=params["use_learned_default"],
            )
            
            trainer = ModelTrainer(
                model=model,
                device="cpu",
                learning_rate=1e-3,
                max_epochs=70,
                patience=12,
                model_name=f"ablation_{exp_name}_seed_{seed}",
                checkpoint_dir=PROJECT_ROOT / "artifacts" / "models",
            )
            trainer.fit(train_loader, val_loader)
            
            # Evaluate on Test C0 (complete)
            test_ds_c0 = MultimodalCancerDataset(
                test_data, y_test, test_sids,
                fixed_condition="C0",
            )
            test_loader_c0 = DataLoader(test_ds_c0, batch_size=64, shuffle=False)
            _, m_c0, _, _ = trainer.evaluate(test_loader_c0)
            c0_scores.append(m_c0["macro_f1"])
            
            # Evaluate on Test C12 (CNV only)
            test_ds_c12 = MultimodalCancerDataset(
                test_data, y_test, test_sids,
                fixed_condition="C12",
            )
            test_loader_c12 = DataLoader(test_ds_c12, batch_size=64, shuffle=False)
            _, m_c12, _, _ = trainer.evaluate(test_loader_c12)
            c12_scores.append(m_c12["macro_f1"])
            
        results[exp_name] = {
            "group": exp["group"],
            "params": params,
            "c0_macro_f1_mean": float(np.mean(c0_scores)),
            "c0_macro_f1_std": float(np.std(c0_scores)),
            "c12_macro_f1_mean": float(np.mean(c12_scores)),
            "c12_macro_f1_std": float(np.std(c12_scores)),
        }
        
        logger.info(
            f"[{exp_name}] C0 F1: {np.mean(c0_scores):.4f} +/- {np.std(c0_scores):.4f} | "
            f"C12 F1: {np.mean(c12_scores):.4f} +/- {np.std(c12_scores):.4f}"
        )
        
    out_path = PROJECT_ROOT / "artifacts" / "metrics" / "ablation_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation study saved to {out_path}")
    return results


if __name__ == "__main__":
    run_ablation_experiments(seeds=[42, 123, 456])
