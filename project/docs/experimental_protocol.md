# Experimental Protocol

## 1. Overview

This document specifies the exact procedures for all experiments, ensuring reproducibility and preventing data leakage.

---

## 2. Data Pipeline

### 2.1 Filtering
1. Load the verified label mapping from `tcga_label_verification.csv`
2. Select samples with labels ∈ {2, 8, 13, 27}
3. For each modality, extract columns corresponding to selected sample IDs
4. Transpose to (samples × features) orientation
5. Verify: 2,263 samples × {3217, 383, 3105, 3139} features

### 2.2 Splitting
1. Input: 2,263 sample IDs with labels
2. Method: `StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)`
3. First split: 85% train+val, 15% test
4. Second split on train+val: `StratifiedShuffleSplit(test_size=0.176, random_state=42)` → 70% train, 15% val (0.176 ≈ 0.15/0.85)
5. Output: `data/splits/split_manifest.csv` with columns: `sample_id, label, cancer_type, split`
6. Verify: No sample in multiple splits; class proportions preserved

### 2.3 Preprocessing (per modality)
1. Fit on training samples only
2. Remove constant features (variance = 0 on training set)
3. Apply StandardScaler (or MinMaxScaler for Methy, pending distribution analysis)
4. Save fitted transformers as pickle files
5. Transform validation and test data using the fitted transformers
6. Never refit on validation/test data

---

## 3. Experiment Execution Order

### Phase 1: Baselines (Stage D)
```
Experiment D1: Single-modality MLP (mRNA)
Experiment D2: Single-modality MLP (miRNA)
Experiment D3: Single-modality MLP (CNV)
Experiment D4: Single-modality MLP (Methy)
Experiment D5: Concatenation MLP (all modalities)
Experiment D6: XGBoost (concatenated features) — optional
```

### Phase 2: Fusion Model (Stage E)
```
Experiment E1: Cross-attention fusion (all modalities, no dropout)
```

### Phase 3: Missing-Modality Experiments (Stage F)
```
Experiment F1: Strategy S1 — Train complete, evaluate all 15 conditions
Experiment F2: Strategy S2 — Train with dropout, evaluate all 15 conditions
Experiment F3: Strategy S3 — Train with curriculum dropout, evaluate all 15 conditions
```

### Phase 4: Ablations (Stage E/F)
```
Experiment A1: Representation dim ablation (D ∈ {64, 128, 256})
Experiment A2: Attention heads ablation (H ∈ {1, 2, 4})
Experiment A3: Modality dropout rate ablation (p ∈ {0.1, 0.2, 0.3, 0.5})
Experiment A4: Learned default vs. zero masking
Experiment A5: Feature selection threshold ablation
```

---

## 4. Per-Experiment Protocol

For each experiment:

1. **Setup:** Load configuration from YAML. Set random seed.
2. **Data:** Load preprocessed data using the canonical split manifest.
3. **Training:** Train for max 100 epochs with early stopping (patience=15).
4. **Validation:** Evaluate on validation set each epoch. Log metrics.
5. **Checkpoint:** Save best model (by validation loss).
6. **Test evaluation:** Load best checkpoint. Evaluate on test set exactly once.
7. **Logging:** Save training curves, validation metrics, test metrics, confusion matrix.
8. **Repeat:** Run with seeds [42, 123, 456, 789, 1024]. Report mean ± std.

---

## 5. Compute Budget

| Experiment Group | Runs | Est. Time (GPU) | Est. Time (CPU) |
|---|---:|---|---|
| Single-modality (4 models × 5 seeds) | 20 | ~1 hour | ~4 hours |
| Concatenation (1 model × 5 seeds) | 5 | ~20 min | ~1.5 hours |
| Fusion complete (1 model × 5 seeds) | 5 | ~30 min | ~2 hours |
| Missing-modality (3 strategies × 5 seeds × 15 conditions eval) | 15 + 225 eval | ~5 hours | ~20 hours |
| Ablations | Variable | ~3 hours | ~12 hours |
| **Total** | | **~10 hours** | **~40 hours** |

If CPU-only, prioritize: D1–D5, E1, F1–F2 with seeds [42, 123, 456].

---

## 6. Result Reporting

Each experiment produces:
- `artifacts/metrics/{experiment_id}_results.json`: All metrics
- `artifacts/metrics/{experiment_id}_confusion.csv`: Confusion matrix
- `artifacts/metrics/{experiment_id}_training_log.csv`: Per-epoch metrics
- `artifacts/models/{experiment_id}_best.pt`: Best model checkpoint
