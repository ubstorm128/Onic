# Missing-Modality Protocol

## 1. Context

The MLOmics Pan-cancer dataset has **no natural missing modalities** — all 8,314 samples have complete data across all four modalities (mRNA, miRNA, CNV, DNA methylation). Therefore, all missing-modality experiments in this project involve **controlled, simulated modality dropout**.

> [!IMPORTANT]
> This study evaluates **robustness under controlled modality dropout**, not recovery from naturally occurring missing data. All claims must be scoped accordingly.

## 2. Definitions

| Term | Meaning |
|---|---|
| **Modality absent at inference** | An entire modality's data is unavailable for a sample. The model must produce a prediction using only the remaining modalities. |
| **Modality dropout during training** | A training-time regularization strategy where entire modalities are randomly masked for each sample in a batch. The true data exists but is hidden from the model. |
| **Biologically measured zero** | A real measurement value of 0.0 (e.g., no expression detected). This is NOT a missing modality. |
| **Missing cell in available modality** | A single value is missing within an otherwise available modality matrix. Not applicable in our dataset. |

## 3. Experimental Conditions

### 3.1 Complete Set (15 conditions)

Each condition specifies which modalities are **available** (✓) vs. **absent** (✗):

| ID | mRNA | miRNA | CNV | Methy | Available Count | Description |
|---|---|---|---|---|---:|---|
| C0 | ✓ | ✓ | ✓ | ✓ | 4 | Complete (reference) |
| C1 | ✗ | ✓ | ✓ | ✓ | 3 | mRNA missing |
| C2 | ✓ | ✗ | ✓ | ✓ | 3 | miRNA missing |
| C3 | ✓ | ✓ | ✗ | ✓ | 3 | CNV missing |
| C4 | ✓ | ✓ | ✓ | ✗ | 3 | Methy missing |
| C5 | ✗ | ✗ | ✓ | ✓ | 2 | mRNA+miRNA missing |
| C6 | ✗ | ✓ | ✗ | ✓ | 2 | mRNA+CNV missing |
| C7 | ✗ | ✓ | ✓ | ✗ | 2 | mRNA+Methy missing |
| C8 | ✓ | ✗ | ✗ | ✓ | 2 | miRNA+CNV missing |
| C9 | ✓ | ✗ | ✓ | ✗ | 2 | miRNA+Methy missing |
| C10 | ✓ | ✓ | ✗ | ✗ | 2 | CNV+Methy missing |
| C11 | ✗ | ✗ | ✗ | ✓ | 1 | Only Methy |
| C12 | ✗ | ✗ | ✓ | ✗ | 1 | Only CNV |
| C13 | ✗ | ✓ | ✗ | ✗ | 1 | Only miRNA |
| C14 | ✓ | ✗ | ✗ | ✗ | 1 | Only mRNA |

### 3.2 Priority Set (if compute-limited)

If full evaluation is too expensive, run at minimum:
- C0 (complete reference)
- C1–C4 (each single modality missing)
- C11–C14 (each single modality alone)
- One two-missing condition (C5)

## 4. Training Strategies

### S1: No Modality Dropout
- Train with all four modalities present for every sample
- At inference, handle missing modalities via zero masking or learned defaults
- Expected: Good performance on C0, poor robustness under missingness

### S2: Random Modality Dropout (p=0.3)
- During training, independently drop each modality with probability p=0.3
- At least one modality is always kept (re-sample if all would be dropped)
- Expected: Slightly lower C0 performance, better robustness

### S3: Curriculum Modality Dropout
- Start training with p=0 (all modalities present)
- Linearly increase p from 0 to 0.5 over the first 50% of epochs
- Hold p=0.5 for the remaining epochs
- Expected: Best of both worlds — learns full representations early, then adapts

## 5. Missing-Modality Handling in the Model

### 5.1 At the Encoder Level
When a modality is absent, its encoder is **not invoked**. Instead:
- Replace with a **learned default embedding** (one per modality, optimized during training)
- This represents "no information available" — distinct from a zero measurement

### 5.2 At the Fusion Level
- The attention mechanism receives a **key_padding_mask** indicating which modality tokens are missing
- Attention weights are computed only over available modalities
- Missing modality tokens do NOT participate in attention computation

### 5.3 At the Pooling Level
- Mean pooling is computed only over available modality tokens
- The normalization divisor counts only present modalities

## 6. Evaluation Protocol for Missing-Modality Experiments

1. Train one model per training strategy (S1, S2, S3)
2. For each trained model, evaluate on the test set under all 15 conditions
3. Use the **same test samples** across all conditions (same seed, same split)
4. Report: macro-F1, balanced accuracy, per-class F1, confusion matrix
5. Compute **degradation** = (metric_C0 − metric_Cx) for each condition Cx
6. Report degradation as both absolute and relative (%) values

## 7. Reporting Requirements

- Table: Macro-F1 for each (strategy × condition) combination
- Heatmap: Degradation matrix (strategies × conditions)
- Bar chart: Per-modality importance (measured by average degradation when that modality is absent)
- Statistical comparison: Are S2/S3 significantly better than S1 under missingness?
