"""
Stratified dataset splitting and split manifest management.
Guarantees zero data leakage and strict reproducibility.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_stratified_splits(
    sample_ids: List[str],
    labels: List[int],
    label_to_cancer: Dict[int, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_path: str | Path = "data/splits/splits.csv",
) -> pd.DataFrame:
    """
    Generate stratified train/validation/test splits.
    
    Split ratios:
        train: ~70%
        val:   ~15%
        test:  ~15%
    
    Returns:
        DataFrame with columns: [sample_id, label, cancer_type, split]
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "label": labels,
        "cancer_type": [label_to_cancer[lbl] for lbl in labels]
    })
    
    # Step 1: Split into train and (val + test)
    val_test_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_seed)
    train_idx, val_test_idx = next(sss1.split(df, df["label"]))
    
    train_df = df.iloc[train_idx].copy()
    val_test_df = df.iloc[val_test_idx].copy()
    
    # Step 2: Split (val + test) into val and test
    test_sub_ratio = test_ratio / val_test_ratio  # 0.15 / 0.30 = 0.50
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_sub_ratio, random_state=random_seed)
    val_sub_idx, test_sub_idx = next(sss2.split(val_test_df, val_test_df["label"]))
    
    val_df = val_test_df.iloc[val_sub_idx].copy()
    test_df = val_test_df.iloc[test_sub_idx].copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Verification checks
    train_set = set(train_df["sample_id"])
    val_set = set(val_df["sample_id"])
    test_set = set(test_df["sample_id"])
    
    assert len(train_set.intersection(val_set)) == 0, "Train and Val splits overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test splits overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test splits overlap!"
    assert len(manifest_df) == len(df), f"Sample count mismatch: {len(manifest_df)} != {len(df)}"
    
    # Save manifest
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(out_p, index=False)
        
    return manifest_df


def load_splits(manifest_path: str | Path = "data/splits/splits.csv") -> pd.DataFrame:
    """Load pre-generated split manifest CSV."""
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Split manifest not found at {p}. Run prepare_dataset.py first.")
    return pd.read_csv(p)
