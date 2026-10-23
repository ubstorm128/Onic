"""
Stage C: Dataset Preparation, Stratified Splitting, and Preprocessing Pipeline
==============================================================================
Loads the raw TCGA omics matrices, filters the 2,263 cohort samples across
the four target cancer classes (BRCA, HNSC, LGG, THCA), generates stratified
train/validation/test splits, fits preprocessors strictly on the training set,
and persists processed arrays and split manifests.
"""

import sys
import csv
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.splitting import create_stratified_splits
from src.data.preprocessor import ModalityPreprocessor


def prepare_dataset() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("prepare_dataset", reports_dir / "prepare_dataset.log")
    logger.info("Starting Stage C Data Preparation Pipeline...")
    
    cfg = load_config()
    data_cfg = cfg["data"]
    set_seed(data_cfg["splitting"]["random_seed"])
    
    raw_dir = Path(data_cfg["raw_data_dir"])
    label_path = raw_dir / "Pan-cancer_label_num.csv"
    
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")
        
    logger.info(f"Loading labels from {label_path}")
    with open(label_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        all_labels = [int(row[0]) for row in reader]
        
    logger.info(f"Read {len(all_labels)} total labels from label file.")
    
    # Read reference sample IDs from Pan-cancer_mRNA.csv header
    ref_mod_file = raw_dir / "Pan-cancer_mRNA.csv"
    with open(ref_mod_file, "r", newline="") as f:
        reader = csv.reader(f)
        ref_header = next(reader)
        all_sample_ids = ref_header[1:]
        
    assert len(all_labels) == len(all_sample_ids) == 8314, (
        f"Length mismatch: {len(all_labels)} labels vs {len(all_sample_ids)} sample IDs"
    )
    
    # Target label set: 2 (BRCA), 8 (HNSC), 13 (LGG), 27 (THCA)
    target_labels = set(data_cfg["label_mapping"].keys())
    
    # Identify indices of target samples in the 8,314 cohort
    target_indices = [i for i, lbl in enumerate(all_labels) if lbl in target_labels]
    assert len(target_indices) == 2263, f"Expected 2263 target samples, found {len(target_indices)}"
    
    cohort_sample_ids = [all_sample_ids[i] for i in target_indices]
    cohort_raw_labels = [all_labels[i] for i in target_indices]
    
    # Verification of class counts
    label_counts = Counter(cohort_raw_labels)
    expected_counts = {2: 757, 8: 506, 13: 506, 27: 494}
    logger.info("Verifying cohort class counts:")
    for lbl, exp_cnt in expected_counts.items():
        c_name = data_cfg["label_mapping"][lbl]
        actual_cnt = label_counts[lbl]
        logger.info(f"  {c_name} (label {lbl}): expected {exp_cnt}, got {actual_cnt}")
        assert actual_cnt == exp_cnt, f"Mismatch for {c_name}: expected {exp_cnt}, got {actual_cnt}"
        
    # Map raw labels (2, 8, 13, 27) to standard target indices (0, 1, 2, 3)
    # 0: BRCA, 1: HNSC, 2: LGG, 3: THCA
    class_name_to_idx = data_cfg["classes"]
    mapped_target_indices = [
        class_name_to_idx[data_cfg["label_mapping"][lbl]]
        for lbl in cohort_raw_labels
    ]
    
    # Step 1: Create Stratified Train/Val/Test Splits (70/15/15)
    logger.info("Generating stratified train/val/test splits (70/15/15)...")
    splits_manifest_path = PROJECT_ROOT / "data" / "splits" / "splits.csv"
    splits_df = create_stratified_splits(
        sample_ids=cohort_sample_ids,
        labels=mapped_target_indices,
        label_to_cancer={v: k for k, v in class_name_to_idx.items()},
        train_ratio=data_cfg["splitting"]["train_ratio"],
        val_ratio=data_cfg["splitting"]["val_ratio"],
        test_ratio=data_cfg["splitting"]["test_ratio"],
        random_seed=data_cfg["splitting"]["random_seed"],
        output_path=splits_manifest_path,
    )
    
    split_summary = splits_df.groupby(["split", "cancer_type"]).size().unstack(fill_value=0)
    logger.info(f"Stratified split summary:\n{split_summary}")
    
    train_ids = set(splits_df[splits_df["split"] == "train"]["sample_id"])
    val_ids = set(splits_df[splits_df["split"] == "val"]["sample_id"])
    test_ids = set(splits_df[splits_df["split"] == "test"]["sample_id"])
    
    train_mask = np.array([sid in train_ids for sid in cohort_sample_ids])
    val_mask = np.array([sid in val_ids for sid in cohort_sample_ids])
    test_mask = np.array([sid in test_ids for sid in cohort_sample_ids])
    
    logger.info(
        f"Split sizes: Train={np.sum(train_mask)} ({np.mean(train_mask):.1%}), "
        f"Val={np.sum(val_mask)} ({np.mean(val_mask):.1%}), "
        f"Test={np.sum(test_mask)} ({np.mean(test_mask):.1%})"
    )
    
    # Step 2: Stream and process each modality matrix
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessors = {}
    modality_metadata = {}
    
    for mod_name, mod_info in data_cfg["modalities"].items():
        file_path = raw_dir / mod_info["filename"]
        logger.info(f"Processing modality {mod_name} from {file_path.name}...")
        
        feature_names = []
        matrix_rows = []
        
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Verify sample alignment
            assert header[1:] == all_sample_ids, f"Sample ID alignment failed for {mod_name}"
            
            for row in reader:
                feat_name = row[0]
                feature_names.append(feat_name)
                # Extract values for target cohort samples
                vals = [float(row[1 + idx]) for idx in target_indices]
                matrix_rows.append(vals)
                
        # matrix_rows is (n_features, 2263), transpose to (2263, n_features)
        X_cohort = np.array(matrix_rows, dtype=np.float32).T
        assert X_cohort.shape == (2263, len(feature_names)), (
            f"Dimension mismatch for {mod_name}: expected (2263, {len(feature_names)}), got {X_cohort.shape}"
        )
        assert len(feature_names) == mod_info["expected_features"], (
            f"Feature count mismatch for {mod_name}: expected {mod_info['expected_features']}, got {len(feature_names)}"
        )
        
        logger.info(f"  {mod_name} raw shape: {X_cohort.shape} (2,263 samples × {len(feature_names)} features)")
        
        # Partition into train, val, test
        X_train = X_cohort[train_mask]
        X_val = X_cohort[val_mask]
        X_test = X_cohort[test_mask]
        
        # Initialize leak-free preprocessor
        prep = ModalityPreprocessor(
            modality_name=mod_name,
            normalization=mod_info["normalization"],
            variance_threshold=data_cfg["preprocessing"]["variance_threshold"]
        )
        # Fit strictly on TRAIN split
        logger.info(f"  Fitting {mod_name} preprocessor exclusively on training split ({X_train.shape[0]} samples)...")
        X_train_proc = prep.fit_transform(X_train, feature_names)
        X_val_proc = prep.transform(X_val)
        X_test_proc = prep.transform(X_test)
        
        dropped_count = prep.original_feature_count - X_train_proc.shape[1]
        logger.info(
            f"  {mod_name} processed shape: train={X_train_proc.shape}, val={X_val_proc.shape}, test={X_test_proc.shape} "
            f"(dropped {dropped_count} zero-variance features)"
        )
        
        # Save fitted preprocessor
        prep.save(processed_dir / f"preprocessor_{mod_name}.joblib")
        preprocessors[mod_name] = prep
        
        # Reconstruct full cohort array in original cohort order for simple indexing
        X_full_proc = np.zeros((2263, X_train_proc.shape[1]), dtype=np.float32)
        X_full_proc[train_mask] = X_train_proc
        X_full_proc[val_mask] = X_val_proc
        X_full_proc[test_mask] = X_test_proc
        
        # Save processed arrays
        np.save(processed_dir / f"{mod_name}_processed.npy", X_full_proc)
        np.save(processed_dir / f"{mod_name}_train.npy", X_train_proc)
        np.save(processed_dir / f"{mod_name}_val.npy", X_val_proc)
        np.save(processed_dir / f"{mod_name}_test.npy", X_test_proc)
        
        modality_metadata[mod_name] = {
            "original_features": prep.original_feature_count,
            "processed_features": X_train_proc.shape[1],
            "dropped_constant_features": dropped_count,
            "normalization": mod_info["normalization"],
            "selected_features": prep.selected_feature_names,
        }
        
    # Save labels and sample IDs
    y_full = np.array(mapped_target_indices, dtype=np.int64)
    np.save(processed_dir / "labels_full.npy", y_full)
    np.save(processed_dir / "labels_train.npy", y_full[train_mask])
    np.save(processed_dir / "labels_val.npy", y_full[val_mask])
    np.save(processed_dir / "labels_test.npy", y_full[test_mask])
    
    with open(processed_dir / "sample_ids_full.json", "w", encoding="utf-8") as f:
        json.dump(cohort_sample_ids, f)
        
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": 2263,
            "train_samples": int(np.sum(train_mask)),
            "val_samples": int(np.sum(val_mask)),
            "test_samples": int(np.sum(test_mask)),
            "modalities": {
                k: {
                    "original_features": v["original_features"],
                    "processed_features": v["processed_features"],
                    "dropped_constant_features": v["dropped_constant_features"],
                    "normalization": v["normalization"],
                }
                for k, v in modality_metadata.items()
            },
            "class_counts": {data_cfg["label_mapping"][k]: v for k, v in label_counts.items()},
            "classes": class_name_to_idx,
        }, f, indent=2)
        
    logger.info("Stage C Data Preparation successfully complete!")
    logger.info(f"Artifacts saved in {processed_dir} and {splits_manifest_path}")


if __name__ == "__main__":
    prepare_dataset()
