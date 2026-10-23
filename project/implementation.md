# Implementation Blueprint

**Project:** A Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities  
**Version:** 1.0  
**Last Updated:** 2026-10-15

---

## 1. Problem Definition

### 1.1 Core Problem

Classify cancer samples into four types (BRCA, HNSC, LGG, THCA) using four omics modalities (mRNA, miRNA, CNV, DNA methylation) while maintaining robustness when one or more modalities are unavailable at inference time.

### 1.2 Research Questions

1. **RQ1:** Does multimodal fusion of four omics modalities improve classification accuracy over single-modality models?
2. **RQ2:** Which fusion strategy (concatenation vs. attention-based) provides better performance?
3. **RQ3:** How does classification performance degrade under systematic missing-modality conditions?
4. **RQ4:** Can training-time modality dropout improve robustness to missing modalities at inference?
5. **RQ5:** Which modalities and features are most informative for each cancer type?

### 1.3 Scope Boundaries

- **In scope:** Four cancer types, four modalities, simulated missingness, explainability
- **Out of scope:** Clinical deployment, survival prediction, drug response, external validation cohorts, natural missingness

---

## 2. Dataset (Verified)

### 2.1 Source

MLOmics Pan-cancer Classification Dataset (TCGA-derived), 8,314 total samples across 32 cancer types.

### 2.2 Target Subset

| Cancer | Label | Samples | TCGA Project |
|---|---:|---:|---|
| BRCA | 2 | 757 | TCGA-BRCA |
| HNSC | 8 | 506 | TCGA-HNSC |
| LGG | 13 | 506 | TCGA-LGG |
| THCA | 27 | 494 | TCGA-THCA |
| **Total** | | **2,263** | |

### 2.3 Modality Dimensions

| Modality | Features | Description |
|---|---:|---|
| mRNA | 3,217 | Gene expression |
| miRNA | 383 | MicroRNA expression |
| CNV | 3,105 | Copy number variation |
| Methy | 3,139 | DNA methylation |
| **Total** | **9,844** | |

### 2.4 Verified Properties

- ✅ All label→project mappings verified via GDC API
- ✅ All modalities share identical sample IDs in identical order
- ✅ No missing values, no non-numeric cells, no non-finite values
- ✅ No duplicate sample IDs, no multi-sample patients
- ✅ No natural missing modalities — all samples have all four modalities
- ⚠ CNV feature column uses `Sample` (capital S); others use `sample`

---

## 3. Data Validation and Quality Control

### 3.1 Completed Checks (Stage A)

See [`docs/dataset_audit.md`](docs/dataset_audit.md) for full results.

### 3.2 Deferred to Preprocessing (Stage C)

1. Per-modality numerical distribution analysis (mean, std, range, skewness, kurtosis)
2. Constant-feature detection across all 2,263 target samples
3. Near-zero-variance feature identification
4. High-correlation feature detection
5. Outlier analysis per modality
6. Scale comparison across modalities (critical for fusion)

---

## 4. Data Preprocessing Strategy

### 4.1 Principles

- Fit all transformations on **training data only**
- Apply fitted transformations to validation and test data
- Process each modality independently (different scales, distributions)
- Preserve feature identifiers throughout for explainability

### 4.2 Per-Modality Pipeline

```
Raw filtered matrix (features × samples)
    → Transpose to (samples × features)
    → Remove constant features (variance = 0 on train)
    → Remove near-zero-variance features (threshold TBD, justified on train)
    → Standardization (fit on train: z-score for mRNA/miRNA/CNV, or min-max for Methy)
    → Optional: Feature selection (variance threshold or mutual information, on train only)
    → Output: (N_samples × K_features) per modality
```

### 4.3 Feature Selection Strategy

**Decision:** Start with variance-based filtering (remove bottom 10% variance features on training set), then evaluate whether further reduction is needed based on computational constraints.

**Rationale:** 
- Total features = 9,844 across modalities. This is manageable for neural networks on modern hardware.
- Aggressive selection (e.g., top 200 per modality) risks discarding discriminative features. We will evaluate this as an ablation.
- Supervised selection (mutual information) must use only training labels.

### 4.4 Normalization Choices

| Modality | Proposed Normalization | Rationale |
|---|---|---|
| mRNA | StandardScaler (z-score) | Expression values span wide range; z-score centers and scales |
| miRNA | StandardScaler (z-score) | Same reasoning as mRNA |
| CNV | StandardScaler (z-score) | Log2 ratios may have heavy tails; z-score is standard |
| Methy | MinMaxScaler or keep as-is | Beta values naturally bounded [0,1]; may not need scaling |

**Decision point:** The methylation scaling choice will be finalized after inspecting actual distributions.

---

## 5. Train/Validation/Test Splitting

### 5.1 Split Ratios

| Split | Proportion | Estimated Samples |
|---|---:|---:|
| Train | 70% | ~1,584 |
| Validation | 15% | ~339 |
| Test | 15% | ~340 |

### 5.2 Splitting Protocol

1. **Stratification:** Stratified by cancer type label to preserve class proportions in each split
2. **Patient grouping:** Not required (verified: no multi-sample patients in target subset)
3. **Random seed:** Fixed at `42` (configurable in `configs/data.yaml`)
4. **Implementation:** `sklearn.model_selection.StratifiedShuffleSplit` or equivalent
5. **Persistence:** Save split assignments as a CSV manifest with columns: `sample_id, label, cancer_type, split`

### 5.3 Anti-Leakage Guarantees

- Split is performed BEFORE any preprocessing, feature selection, or model training
- Validation set is used for model selection and hyperparameter tuning
- Test set is used ONLY for final evaluation — never for tuning

---

## 6. Baseline Models

### 6.1 Single-Modality MLP Baselines

For each modality independently:

```
Input (K features)
    → Linear(K, 256) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256, 128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128, 4)   → Softmax
```

**Purpose:** Establish per-modality classification capability. Determines which modalities are independently informative.

### 6.2 Early Fusion (Concatenation) Baseline

```
Concatenate all modalities: Input (3217 + 383 + 3105 + 3139 = 9844 features)
    → Linear(9844, 512) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(512, 256)  → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256, 128)  → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128, 4)    → Softmax
```

**Purpose:** Measure whether simple concatenation improves over single modalities. This is the multimodal baseline to beat.

### 6.3 Conventional Tabular Baseline (Optional)

A gradient-boosted tree (XGBoost or LightGBM) on concatenated features.

**Purpose:** Neural networks are not always superior for tabular data. This checks whether the problem even benefits from deep learning.

---

## 7. Proposed Multimodal Fusion Architecture

### 7.1 Overview

```
mRNA  → Encoder_mRNA  → z_mRNA  (D-dim) ─┐
miRNA → Encoder_miRNA → z_miRNA (D-dim) ─┤
CNV   → Encoder_CNV   → z_CNV   (D-dim) ─┼─→ Fusion Module → h_fused (D-dim) → Classifier → 4 classes
Methy → Encoder_Methy → z_Methy (D-dim) ─┘
```

### 7.2 Modality-Specific Encoders

Each encoder reduces its modality to a shared representation dimension `D`:

```python
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        # Linear(input_dim, hidden_dim) → BN → ReLU → Dropout
        # Linear(hidden_dim, output_dim) → BN → ReLU → Dropout
```

### 7.3 Fusion Module

**Primary approach: Multi-Head Cross-Attention Fusion**

```
Stack modality embeddings: [z_mRNA, z_miRNA, z_CNV, z_Methy] → (4, D)
Add learned modality-type embeddings
Apply multi-head self-attention across the 4 modality tokens
Pool (mean or CLS token) → h_fused (D)
```

**Justification for attention:** With only 4 tokens (one per modality), the attention mechanism is lightweight (no quadratic scaling concern). It enables the model to learn **which modalities to attend to** for each sample, which directly supports the missing-modality robustness goal — when a modality is absent, attention should redistribute to available modalities.

**Alternative (ablation):** Simple mean/max pooling of encoder outputs, gated fusion.

### 7.4 Missing-Modality Handling

When a modality is absent:
1. Replace the modality's encoder output with a **learned default embedding** (not zeros)
2. Add a **modality availability mask** to the attention mechanism
3. Mask out absent modalities so attention is computed only over present ones

### 7.5 Classification Head

```
h_fused (D) → Linear(D, 64) → ReLU → Dropout → Linear(64, 4)
```

### 7.6 Hyperparameter Defaults

| Parameter | Value | Notes |
|---|---|---|
| Encoder hidden dim | 256 | |
| Representation dim D | 128 | Shared across modalities |
| Attention heads | 4 | One per modality conceptually |
| Attention layers | 1 | Minimal; ablate depth |
| Dropout rate | 0.3 | |
| Learning rate | 1e-3 | With cosine annealing |
| Batch size | 64 | |
| Max epochs | 100 | With early stopping (patience=15) |
| Optimizer | AdamW | Weight decay 1e-4 |
| Loss | Cross-entropy | Optionally class-weighted |

---

## 8. Missing-Modality Simulation Strategy

### 8.1 Key Distinction

The source data has **no natural missing modalities**. All experiments involve **controlled modality dropout** to evaluate robustness. This must be stated clearly in any reporting.

### 8.2 Experimental Conditions

#### Evaluation conditions (15 total):

| Condition | mRNA | miRNA | CNV | Methy | Description |
|---|---|---|---|---|---|
| C0 | ✓ | ✓ | ✓ | ✓ | Complete (reference) |
| C1 | ✗ | ✓ | ✓ | ✓ | mRNA missing |
| C2 | ✓ | ✗ | ✓ | ✓ | miRNA missing |
| C3 | ✓ | ✓ | ✗ | ✓ | CNV missing |
| C4 | ✓ | ✓ | ✓ | ✗ | Methy missing |
| C5 | ✗ | ✗ | ✓ | ✓ | mRNA+miRNA missing |
| C6 | ✗ | ✓ | ✗ | ✓ | mRNA+CNV missing |
| C7 | ✗ | ✓ | ✓ | ✗ | mRNA+Methy missing |
| C8 | ✓ | ✗ | ✗ | ✓ | miRNA+CNV missing |
| C9 | ✓ | ✗ | ✓ | ✗ | miRNA+Methy missing |
| C10 | ✓ | ✓ | ✗ | ✗ | CNV+Methy missing |
| C11 | ✗ | ✗ | ✗ | ✓ | Only Methy |
| C12 | ✗ | ✗ | ✓ | ✗ | Only CNV |
| C13 | ✗ | ✓ | ✗ | ✗ | Only miRNA |
| C14 | ✓ | ✗ | ✗ | ✗ | Only mRNA |

#### Training strategies:

| Strategy | Description |
|---|---|
| S1: No dropout | Train with all modalities; evaluate under missingness (fragile baseline) |
| S2: Modality dropout | During training, randomly drop each modality with probability p=0.3 |
| S3: Curriculum dropout | Start with all modalities, gradually increase dropout probability |

### 8.3 Critical Distinctions

| Concept | Meaning |
|---|---|
| Modality absent at inference | Entire modality unavailable for a sample |
| Modality dropout during training | Regularization technique; modality randomly masked per batch |
| Biologically measured zero | A real measurement of zero expression/methylation |
| Missing cell within a modality | A single value missing in an otherwise available matrix |

These are **not equivalent** and must not be confused.

---

## 9. Training and Optimization

### 9.1 Training Loop

```
For each epoch:
    For each batch:
        1. Load batch (samples × features for each modality)
        2. Apply modality dropout mask (if training strategy S2/S3)
        3. Forward pass through encoders → fusion → classifier
        4. Compute cross-entropy loss
        5. Backward pass and optimizer step
    
    Evaluate on validation set (complete modalities)
    Check early stopping criterion
    Save checkpoint if best validation loss
```

### 9.2 Early Stopping

- Monitor: validation loss
- Patience: 15 epochs
- Restore best model weights

### 9.3 Learning Rate Schedule

- Cosine annealing from initial LR to 1e-6 over max epochs
- Warmup: first 5 epochs (linear warmup)

### 9.4 Hardware Considerations

- **Target:** Single GPU (CUDA) or CPU fallback
- **Memory estimate:** 2,263 samples × 9,844 features × 4 bytes ≈ 85 MB per full matrix. Easily fits in RAM.
- **Batch processing:** 64 samples per batch. No memory concern for this dataset size.

---

## 10. Evaluation Metrics

### 10.1 Primary Metrics

| Metric | Description | Aggregation |
|---|---|---|
| Macro-F1 | F1 averaged equally across classes | Primary metric |
| Balanced Accuracy | Average of per-class recall | Secondary metric |
| Overall Accuracy | Proportion correctly classified | Report but do not optimize |

### 10.2 Per-Class Metrics

For each cancer type: precision, recall, F1-score.

### 10.3 Confusion Matrix

4×4 matrix for each experimental condition.

### 10.4 Statistical Reporting

- Run each experiment with 5 random seeds
- Report mean ± standard deviation
- Use Wilcoxon signed-rank test for pairwise model comparisons (paired across seeds)

### 10.5 Reporting Protocol

- Validation results: used for model selection and hyperparameter tuning
- Test results: reported ONCE per final model configuration
- Never report only the best seed

---

## 11. Explainability Strategy

### 11.1 Modality-Level Importance

1. **Permutation importance:** Shuffle each modality's features and measure performance drop
2. **Ablation importance:** Remove each modality and compare to complete input
3. **Attention analysis (if applicable):** Report average attention weights per modality, with caveats

### 11.2 Feature-Level Importance

1. **Integrated gradients:** Attribution of predictions to individual features
2. **SHAP (if computationally feasible):** Kernel SHAP or Deep SHAP for feature-level explanations
3. **Top-K feature ranking:** Identify most important features per modality and per class

### 11.3 Local Explanations

- Per-sample feature attribution for selected correctly/incorrectly classified examples
- Comparison of explanations across missingness conditions

### 11.4 Caveats

- Attention weights ≠ explanations (self-attention weights are not guaranteed to reflect importance)
- No causal claims from any explanation method
- Explainability computed on test set but NEVER used for model selection

---

## 12. Ablation Experiments

| Ablation | Question |
|---|---|
| Single vs. multi-modality | Does fusion help? |
| Concatenation vs. attention fusion | Does attention help? |
| With/without modality dropout training | Does dropout help robustness? |
| Feature selection thresholds | How sensitive is performance to feature count? |
| Representation dimension D | Effect of bottleneck size (64, 128, 256) |
| Number of attention heads | Effect on fusion quality (1, 2, 4) |
| Learned default vs. zero masking | Best strategy for missing modalities |

---

## 13. Statistical Reporting and Reproducibility

### 13.1 Fixed Seeds

| Component | Seed |
|---|---|
| Data splitting | 42 |
| Model initialization | [42, 123, 456, 789, 1024] |
| Modality dropout | Same as model seed |

### 13.2 Reproducibility Artifacts

For each experiment:
- Configuration YAML
- Split manifest CSV
- Model checkpoint (.pt)
- Training log (loss, metrics per epoch)
- Evaluation results JSON
- Random seed used

---

## 14. Hardware and Memory Constraints

| Resource | Estimate |
|---|---|
| Dataset (filtered, all modalities) | ~380 MB on disk, ~85 MB in memory |
| Model parameters | ~3–5M (estimated) |
| Training memory (batch=64) | ~500 MB GPU, ~2 GB with gradients |
| Total training time | ~30 min on GPU, ~2–4 hours on CPU |
| Missing-modality experiments (15×3) | ~22.5 hours GPU, ~6–12 days CPU |

**Mitigation:** Reduce experiment matrix if CPU-only. Prioritize: complete, single-missing, selected two-missing.

---

## 15. Risks, Limitations, and Fallback Strategies

| Risk | Mitigation |
|---|---|
| GPU unavailable | CPU fallback; reduce experiment scope |
| Overfitting on small dataset | Dropout, early stopping, weight decay |
| Attention overhead unjustified | Compare to simple mean pooling |
| Feature selection discards signal | Ablation comparing different thresholds |
| Class imbalance affects metrics | Class-weighted loss; report macro-F1 and balanced accuracy |
| Explainability misleading | Multiple methods; clearly state limitations |

---

## 16. Acceptance Criteria

### Stage C Gate
- [ ] Filtered data has exactly 2,263 samples across all modalities
- [ ] Class counts: BRCA=757, HNSC=506, LGG=506, THCA=494
- [ ] Train/val/test splits are stratified and reproducible
- [ ] No sample appears in multiple splits
- [ ] Preprocessing fitted on training data only

### Stage D Gate
- [ ] Single-modality models train and evaluate without errors
- [ ] Concatenation baseline trains and evaluates without errors
- [ ] All models use identical splits and evaluation protocol
- [ ] Results are logged and reproducible

### Stage E Gate
- [ ] Fusion model trains with correct tensor shapes
- [ ] Gradients flow through all components
- [ ] Checkpoint save/load works correctly
- [ ] Performance is reported on validation set

### Stage F Gate
- [ ] All 15 missingness conditions are evaluated
- [ ] At least 2 training strategies are compared
- [ ] Missing modalities use learned defaults (not fabricated data)
- [ ] Results are tabulated per condition

### Stage G Gate
- [ ] Modality importance is computed and reported
- [ ] Feature importance is computed for at least the top modality
- [ ] All results trace to saved configs, seeds, and checkpoints
- [ ] No test set was used for model selection
