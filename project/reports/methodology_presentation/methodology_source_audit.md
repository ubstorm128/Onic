# Technical Methodology Source Audit & Code Verification

**Project Title:** A Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities  
**Audit Purpose:** Traceability matrix linking every presentation slide, claim, and hyperparameter to its underlying source file, checkpoint, or test execution.

---

## 1. Technical Claims Traceability Matrix

| Slide / Topic | Technical Claim | Source File / Artifact | Exact Code / Line Reference | Verification Status |
|:---|:---|:---|:---|:---:|
| **Slide 1: Overview** | 5 core stages from cohort selection to paired statistical testing | `PROJECT_STATUS.md`, `scripts/run_statistical_tests.py` | Full pipeline execution | **VERIFIED FROM CODE** |
| **Slide 2: Cohort** | Total cohort: 2,263 unique patients; 0 multi-sample clones | `data/splits/splits.csv`, `scripts/verify_patient_leakage.py` | `verify_patient_leakage.py:L25-L65` | **VERIFIED FROM CODE** |
| **Slide 2: Cohort** | 4 Cancer types: BRCA (757), HNSC (506), LGG (506), THCA (494) | `configs/data.yaml:L10-L22` | `class_distribution.csv` | **VERIFIED FROM CODE** |
| **Slide 2: Modalities** | 4 Omics: mRNA (3217), miRNA (383), CNV (3105), Methy (3139) | `configs/data.yaml:L24-L45` | `data/processed/*_processed.npy` | **VERIFIED FROM CODE** |
| **Slide 3: Splitting** | 70% Train (1584), 15% Val (339), 15% Test (340) | `data/splits/splits.csv` | `splits.csv` counts | **VERIFIED FROM CODE** |
| **Slide 3: Anti-Leakage** | Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅ | `tests/test_pipeline.py:L27-L41` | `test_splits_no_leakage()` | **VERIFIED (TEST PASSED)** |
| **Slide 3: Preprocessing** | StandardScaler on mRNA, miRNA, CNV; MinMaxScaler on Methy | `src/data/preprocessor.py:L20-L70` | `configs/data.yaml:L24-L45` | **VERIFIED FROM CODE** |
| **Slide 3: Preprocessing** | Train-only fitting; zero-variance cutoff; 9,844 features kept | `src/data/preprocessor.py:L36-L70` | `preprocessor.py:fit()` | **VERIFIED FROM CODE** |
| **Slide 4: Architecture** | Two-layer non-linear MLPs (K -> 256 -> 128) with BN, ReLU, Dropout | `src/models/fusion.py:L20-L60` | `ModalityEncoder:forward()` | **VERIFIED FROM CODE** |
| **Slide 4: Architecture** | Modality type embeddings E_m ∈ R^128, default tokens θ_m ∈ R^128 | `src/models/fusion.py:L101-L115` | `nn.Parameter(...)` | **VERIFIED FROM CODE** |
| **Slide 4: Architecture** | Multi-Head Self-Attention over tokens (H=4, D=128) with key padding | `src/models/fusion.py:L117-L135` | `nn.MultiheadAttention` | **VERIFIED FROM CODE** |
| **Slide 4: Architecture** | Clarification: Self-attention over tokens, not cross-attention | `src/models/fusion.py:L202-L210` | `query=tokens, key=tokens, value=tokens` | **VERIFIED FROM CODE** |
| **Slide 4: Architecture** | Mask-aware mean pooling: h_fused = ∑ M_m z'_m / max(1, ∑ M_m) | `src/models/fusion.py:L217-L223` | `h_fused = masked_tokens.sum(1) / mask_sum` | **VERIFIED FROM CODE** |
| **Slide 4: Architecture** | Classifier: Linear(128 -> 64) -> ReLU -> Dropout -> Linear(64 -> 4) | `src/models/fusion.py:L136-L141` | `self.classifier` | **VERIFIED FROM CODE** |
| **Slide 5: Missingness** | 15 predefined conditions C0 through C14 | `src/data/dataset.py:L24-L40` | `EVAL_CONDITIONS` | **VERIFIED FROM CODE** |
| **Slide 5: Missingness** | Numerical invariance: altering masked inputs causes 0.0 logit change | `tests/test_pipeline.py:L112-L162` | `test_masked_modality_numerical_invariance` | **VERIFIED (TEST PASSED)** |
| **Slide 5: Missingness** | All-missing mask [0,0,0,0] numerical safeguard outputs finite prior | `tests/test_pipeline.py:L205-L235` | `test_all_missing_numerical_safeguard` | **VERIFIED (TEST PASSED)** |
| **Slide 6: Training** | Strategy S1 (p=0.0) vs Strategy S2 (dropout p=0.3) | `scripts/run_fusion.py:L68-L75` | `MultimodalCancerDataset` | **VERIFIED FROM CODE** |
| **Slide 6: Training** | AdamW (lr=1e-3, wd=1e-4), Cosine Annealing, Batch size=64 | `src/training/trainer.py:L51-L63` | `ModelTrainer:__init__()` | **VERIFIED FROM CODE** |
| **Slide 6: Training** | 5 fixed random seeds: [42, 123, 456, 789, 1024] | `configs/experiment.yaml:L15-L20` | `seeds: [42, 123, 456, 789, 1024]` | **VERIFIED FROM CODE** |
| **Slide 6: Training** | Early stopping patience = 15 on validation loss | `src/training/trainer.py:L180-L205` | `epochs_no_improve >= patience` | **VERIFIED FROM CODE** |
| **Slide 7: Evaluation** | Complete reference C0: S1 = 0.9956 ± 0.0015, S2 = 0.9950 ± 0.0012 | `reports/per_seed_missing_modality_results.csv` | Condition C0 rows | **VERIFIED FROM DATA** |
| **Slide 7: Evaluation** | Isolated CNV C12: S1 = 0.6561 ± 0.0892, S2 = 0.7412 ± 0.0476 | `reports/per_seed_missing_modality_results.csv` | Condition C12 rows | **VERIFIED FROM DATA** |
| **Slide 7: Evaluation** | Paired gain: +0.0852 (+8.52%), Cohen's d = 0.7425, t=1.6603, p=0.1722 | `artifacts/metrics/statistical_significance_tests.json` | `paired_missing_modality_comparisons` | **VERIFIED FROM DATA** |
| **Slide 7: Evaluation** | Statistical non-significance at α=0.05 honestly reported | `reports/statistical_significance_report.md` | Section 1 table & narrative | **VERIFIED FROM REPORT** |
| **Slide 7: Evaluation** | Baselines D5 & D6 evaluated on complete inputs only (LightGBM 0.9973) | `reports/statistical_significance_report.md` | Section 2 table & note | **VERIFIED FROM REPORT** |
| **Slide 8: Timeline** | All 7 implementation milestones completed; 59 checkpoints intact | `artifacts/models/*.pt` | File listing (59 .pt checkpoints) | **VERIFIED FROM ARTIFACTS** |
| **Slide 8: Future Work** | External cohorts (CPTAC/ICGC) and baseline imputation unexecuted | `PROJECT_STATUS.md`, `reports/research_paper_draft.md` | Limitations section | **QUALIFIED AS PENDING** |
| **Slide 9: Summary** | 4 core contributions synthesized without unsupported clinical claims | `reports/methodology_presentation/implementation_methodology.md` | Section 8 & Conclusion | **VERIFIED FROM DOCS** |

---

## 2. Checkpoint and Metric Artifact Verification

1. **Checkpoints Preserved (59 Files, Read-Only):**
   - Proposed S1 Checkpoints (5 seeds): `artifacts/models/fusion_s1_complete_seed_{42,123,456,789,1024}_best.pt`
   - Proposed S2 Checkpoints (5 seeds): `artifacts/models/fusion_s2_dropout_seed_{42,123,456,789,1024}_best.pt`
   - Unimodal Baseline Checkpoints (20 files): `artifacts/models/mlp_{mRNA,miRNA,CNV,Methy}_seed_*_best.pt`
   - Early Fusion Checkpoints (5 seeds): `artifacts/models/early_fusion_seed_*_best.pt`
   - Ablation Checkpoints (24 files): `artifacts/models/ablation_*_best.pt`
2. **Evaluations NPZ:**
   - Saved test predictions across all 15 conditions and 5 seeds: `artifacts/metrics/s1_s2_test_predictions.npz`
   - Test evaluation probability matrices and fused representations: `reports/presentation_figures/data/extracted_test_evaluations.npz`
3. **Unit Tests Executed:**
   - Command: `pytest tests/test_pipeline.py -v`
   - Result: 8 passed in 13.87 seconds (Exit Code 0).

---

## 3. Discrepancy & Qualification Audit

1. **Self-Attention vs. Cross-Attention Terminology:**
   - *Observation:* The class name in `src/models/fusion.py` is `MultimodalCrossAttentionClassifier`.
   - *Code Audit:* Lines 202–210 execute `self.attention(query=tokens, key=tokens, value=tokens)`.
   - *Resolution:* Accurately documented in Slide 4 and the methodology report that the underlying mathematical operation is **Multi-Head Self-Attention over a sequence of 4 modality tokens**, where tokens symmetrically attend across each other.
2. **Training Trajectory Logs:**
   - *Observation:* Continuous 100-epoch training loss and accuracy history arrays were not saved to disk during the original training script.
   - *Resolution:* Transparently documented in Slide 1, Slide 6, and Figure A5 that values reflect verified checkpoint optimal validation states without fabricating synthetic trajectories.
3. **Baseline Comparison Scope:**
   - *Observation:* Baseline models D5 (Early Fusion MLP) and D6 (LightGBM) do not have missing-modality evaluation arrays.
   - *Resolution:* Explicitly qualified on Slide 7, Slide 8, and Figure B3 that D5 and D6 were evaluated **strictly on complete inputs (C0)**, and that LightGBM achieves the highest complete-data score ($0.9973$).
4. **Condition C12 Statistical Significance:**
   - *Observation:* Modality dropout achieves an $+8.52\%$ Macro-F1 improvement on isolated CNV ($0.6561 \to 0.7412, d=0.7425$).
   - *Resolution:* Paired $t$-test yields $p=0.1722$ and 95% CI crosses zero ($[-0.0573, +0.2276]$). Accurately qualified as an empirically substantial effect that does **not** attain nominal statistical significance at $\alpha = 0.05$ due to small degrees of freedom ($df=4$).
