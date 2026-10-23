# Decision Log

This document records key research and implementation decisions, their rationale, considered alternatives, and evidence.

---

## DEC-001: Target Cancer Types Selection

**Date:** 2026-10-15  
**Decision:** Use BRCA, HNSC, LGG, THCA (labels 2, 8, 13, 27)

**Rationale:**
- These four types span different organ systems and molecular profiles
- Reasonable sample sizes (494–757 per class)
- Verified 1:1 mapping between numeric labels and TCGA projects

**Alternatives considered:**
- Using all 32 cancer types: Rejected — too many classes for the primary research question (missing modalities), would dilute the focus
- Using only 2 classes: Rejected — too simple, wouldn't demonstrate multi-class capability

**Evidence:** Label-to-project verification via GDC API (see `tcga_label_crosstab.csv`)

---

## DEC-002: Simulated vs. Natural Missing Modalities

**Date:** 2026-10-15  
**Decision:** Simulate missing modalities via controlled dropout. Describe as "robustness under controlled modality dropout."

**Rationale:**
- All 8,314 samples in the MLOmics dataset have complete data across all four modalities
- No evidence of natural missingness in the source data
- Simulated missingness allows controlled experimentation

**Alternatives considered:**
- Claiming natural missingness: Rejected — not supported by data
- Creating artificial missingness correlated with class: Interesting but out of scope

**Evidence:** Dataset audit confirms 0 missing values across all modalities

---

## DEC-003: Feature Selection Strategy

**Date:** 2026-10-15  
**Decision:** Start with variance-based filtering (remove bottom 10% variance on training set), evaluate further reduction as ablation

**Rationale:**
- Total features (9,844) are manageable for neural networks
- Aggressive selection risks discarding discriminative features
- Variance filtering is unsupervised, avoids label leakage
- Can be refined based on initial results

**Alternatives considered:**
- Fixed top-200 per modality: Rejected — arbitrary, no scientific justification
- Mutual information: Valid but supervised — must be done within training set only
- PCA: Could complement but adds complexity; evaluate as ablation

---

## DEC-004: Fusion Architecture Choice

**Date:** 2026-10-15  
**Decision:** Multi-head self-attention over modality tokens (4 tokens, one per modality)

**Rationale:**
- With only 4 tokens, attention is extremely lightweight (no quadratic scaling issue)
- Attention can naturally handle missing tokens via masking
- Learned modality-type embeddings allow the model to distinguish modalities
- Directly supports the missing-modality research question

**Alternatives considered:**
- Simple concatenation: Included as baseline — cannot handle missing modalities without imputation
- Gated fusion: Valid alternative — will include as ablation if time permits
- Cross-attention (queries from one modality, keys/values from another): More complex, 6 pairwise combinations — harder to scale

---

## DEC-005: Train/Val/Test Split Ratios

**Date:** 2026-10-15  
**Decision:** 70% train / 15% val / 15% test, stratified by cancer type

**Rationale:**
- Standard split for moderate-sized datasets
- 15% test (~340 samples) provides reasonable statistical power for evaluation
- Stratification ensures class proportions are preserved
- No patient-level grouping needed (verified: no multi-sample patients)

**Alternatives considered:**
- 80/10/10: Less validation/test data
- K-fold cross-validation: More robust estimates but 5× training cost; use 5-seed repetition instead

---

## DEC-006: Learned Default Embeddings for Missing Modalities

**Date:** 2026-10-15  
**Decision:** Use learned (trainable) default embeddings rather than zero vectors for absent modalities

**Rationale:**
- Zero vectors have a specific meaning in the representation space (they're a valid point)
- A learned embedding can represent "no information" more appropriately
- The model can learn to distinguish "missing" from "zero-valued"

**Alternatives considered:**
- Zero masking: Simpler but conflates missing data with zero measurements
- Mean embedding (from training set): Fixed, not end-to-end optimized
- Will compare learned vs. zero in ablation experiment A4

---

## DEC-007: Evaluation Metric Priority

**Date:** 2026-10-15  
**Decision:** Primary metric: Macro-F1. Secondary: Balanced Accuracy.

**Rationale:**
- Macro-F1 treats all classes equally regardless of sample count
- Appropriate for moderate imbalance (BRCA has 1.5× THCA)
- Balanced accuracy provides complementary perspective
- Overall accuracy is reported but deprioritized (biased toward BRCA)
