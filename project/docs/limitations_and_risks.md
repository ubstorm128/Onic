# Limitations and Risks

## 1. Dataset Limitations

### 1.1 Limited Cancer Types
Only 4 of 32 available cancer types are used. Results may not generalize to other cancers, especially those with different molecular profiles.

### 1.2 No External Validation
All data comes from a single source (TCGA via MLOmics). No independent cohort is used for external validation. Reported performance may be optimistic.

### 1.3 Pre-Processed Features
The MLOmics dataset provides pre-selected features (3,217 mRNA genes, not the full ~20,000 gene transcriptome). The feature selection criteria applied by the dataset authors are not fully documented.

### 1.4 Sample Size
2,263 samples across 4 classes is modest for deep learning. Overfitting is a significant risk, mitigated by dropout, early stopping, and weight decay, but not eliminated.

### 1.5 TCGA Cohort Bias
TCGA samples are predominantly from North American institutions and may not represent global patient populations. Tumor types, staging, and demographics may be biased.

## 2. Methodological Limitations

### 2.1 Simulated Missing Modalities
All missing-modality experiments use **controlled simulation**, not naturally occurring missingness. In clinical practice, missingness may correlate with patient characteristics (not-at-random), which our simulation does not capture.

### 2.2 No Clinical Validation
This is a computational study. No clinical outcomes are predicted or validated. The model is NOT suitable for clinical decision-making.

### 2.3 Explainability Limitations
- Feature attributions are model-specific and do not imply biological causation
- Attention weights are not reliable indicators of importance
- No experimental validation of identified biomarkers

### 2.4 Statistical Power
With 5 random seeds, statistical comparisons have limited power. Results should be interpreted as indicative, not definitive.

### 2.5 Single Architecture
We evaluate one primary fusion architecture. Other approaches (graph neural networks, kernel methods, variational autoencoders) may perform differently.

## 3. Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Overfitting | Inflated performance | High | Dropout, early stopping, weight decay, validation monitoring |
| Data leakage | Invalid results | Medium | Strict split-before-preprocess protocol, automated tests |
| GPU unavailable | Slow experiments | Medium | CPU fallback, reduced experiment scope |
| Memory overflow | Crashes | Low | Dataset fits in ~85 MB; batch processing |
| Non-determinism | Irreproducible results | Low | Fixed seeds, deterministic mode |

## 4. Ethical and Responsible-Use Statement

### 4.1 Not a Diagnostic Tool
This framework is a **research prototype**. It has NOT been validated for clinical use, regulatory approval, or medical decision-making. Using it for patient diagnosis would be irresponsible and potentially harmful.

### 4.2 No Individual Patient Claims
Results are reported at the population level. The model's predictions for individual patients should not be interpreted as diagnoses.

### 4.3 Bias Considerations
The TCGA dataset has known demographic and institutional biases. Model performance may vary across patient subgroups not represented in the training data.

### 4.4 Data Privacy
All data used is from publicly available, de-identified TCGA datasets. No private patient data is used.

## 5. What This Project Does NOT Claim

- ❌ Clinical readiness or diagnostic capability
- ❌ Causal discovery of cancer mechanisms
- ❌ Biological validation of feature importance
- ❌ Superiority over all existing methods
- ❌ Generalization to cancers not studied
- ❌ Robustness to naturally occurring (non-random) missing data
- ❌ Bidirectional concept-level explainability (unless explicitly implemented and validated)
