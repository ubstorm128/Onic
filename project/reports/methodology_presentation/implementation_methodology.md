# Research Methodology and Implementation Plan

**Project Title:** A Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities  
**Author:** Major Project Research Candidate  
**Document Purpose:** College Major Project Technical Review & Methodology Defense  
**Status:** Audited, Verified, and Complete (59 PyTorch Checkpoints Intact)  

---

## 1. Research Objective and Problem Formulation

High-throughput multi-omics integration provides unprecedented resolution for cancer subtyping, molecular diagnostics, and targeted therapeutic planning. However, clinical translational utility is severely constrained by the *missing modality problem*: in real-world diagnostic workflows, patients rarely receive all profiling assays simultaneously due to limited biopsy tissue volumes, archival formalin-fixed paraffin-embedded (FFPE) sample degradation, institutional budget disparities, and sequencing assay failure.

Conventional multimodal deep learning frameworks suffer from two fatal vulnerabilities:
1. **Rigid Dimensionality Requirement:** Architectures relying on feature concatenation crash or output corrupted representations when one or more modalities are absent.
2. **Inter-Modality Co-Adaptation:** When trained exclusively on complete data, networks over-rely on high-signal assays (e.g., transcriptomic mRNA or epigenetic methylation) and fail to extract self-sufficient representations from noisy structural assays (e.g., genomic copy number variations). When a noisy modality is isolated at test time, performance collapses catastrophically.

### Core Mathematical Objective
We formulate the problem as a four-class classification task under arbitrary missingness across four biological modalities:
$$\mathcal{M} = \{\text{mRNA}, \text{miRNA}, \text{CNV}, \text{Methy}\}$$
Given a patient sample $i$, the model receives a dictionary of raw feature vectors $x_i = \{x_{i,m} \in \mathbb{R}^{K_m} \mid m \in \mathcal{M}\}$ and a binary availability indicator vector:
$$M_i = [M_{i,\text{mRNA}}, M_{i,\text{miRNA}}, M_{i,\text{CNV}}, M_{i,\text{Methy}}] \in \{0, 1\}^4$$
where $M_{i,m} = 1$ denotes modality presence and $M_{i,m} = 0$ denotes assay absence. The objective is to learn a parameterized mapping $f_\Theta(x_i, M_i) \to \hat{y}_i \in \{0, 1, 2, 3\}$ that maximizes classification Macro-F1 across all $2^4 - 1 = 15$ non-empty availability conditions.

---

## 2. Dataset Harmonization and Cohort Verification

### 2.1 Study Population and Organ Distribution
The benchmark cohort is derived from the TCGA MLOmics Pan-Cancer repository (Zenodo DOI: 10.5281/zenodo.28729127). Following verification against canonical TCGA registers, four distinct cancer types were selected:
- **BRCA (Breast Invasive Carcinoma):** $N = 757$ patients ($33.5\%$) — Label 0
- **HNSC (Head and Neck Squamous Cell Carcinoma):** $N = 506$ patients ($22.4\%$) — Label 1
- **LGG (Brain Lower Grade Glioma):** $N = 506$ patients ($22.4\%$) — Label 2
- **THCA (Thyroid Carcinoma):** $N = 494$ patients ($21.8\%$) — Label 3
- **Total Cohort Size:** $N = 2,263$ patients.

### 2.2 Four Biological Profiling Assays (9,844 Features)
1. **mRNA Expression (3,217 genes):** RNA-Seq RSEM normalized gene counts capturing downstream functional pathway activations.
2. **miRNA Regulatory Expression (383 microRNAs):** miRNA-Seq normalized counts reflecting post-transcriptional gene silencing mechanisms.
3. **Copy Number Variation — CNV (3,105 genomic regions):** GISTIC2 thresholded segment values ($-2, -1, 0, +1, +2$) reflecting deletions and amplifications.
4. **DNA Methylation (3,139 CpG probes):** Illumina Infinium HumanMethylation450 platform $\beta$-values $\in [0, 1]$ reflecting promoter hyper/hypo-methylation.

### 2.3 Patient-Level Uniqueness Audit
Executed verification via `scripts/verify_patient_leakage.py` established:
- Exactly 2,263 unique sample IDs correspond to 2,263 unique 12-character TCGA patient barcodes.
- Exactly zero multi-sample or multi-aliquot patients exist in the four-cancer cohort.
- Zero NaN or null entries exist across the 9,844 feature dimensions.

---

## 3. Data Preprocessing and Anti-Leakage Partitioning

### 3.1 Stratified Partitioning Protocol
The 2,263 patients were partitioned into stratified subsets with fixed seed 42:
- **Training Partition (70.0%):** $N = 1,584$ patients (BRCA: 530, HNSC: 354, LGG: 354, THCA: 346)
- **Validation Partition (15.0%):** $N = 339$ patients (BRCA: 113, HNSC: 76, LGG: 76, THCA: 74)
- **Held-Out Test Partition (15.0%):** $N = 340$ patients (BRCA: 114, HNSC: 76, LGG: 76, THCA: 74)

### 3.2 Anti-Leakage Preprocessing Protocol
To prevent data contamination, all transformation parameters were fitted exclusively on the 1,584 training samples:
- `StandardScaler` (z-score normalization, mean $= 0$, variance $= 1$): Applied to mRNA, miRNA, and CNV.
- `MinMaxScaler` (bounded $[0, 1]$ linear scaling): Applied to DNA Methylation to preserve natural biological $\beta$-value bounds.
- `VarianceThreshold(0.0)`: Applied on training split to eliminate strictly constant features. All 9,844 features exhibited non-zero training variance and were retained.
- Fitted preprocessors were saved using `joblib` and applied to validation and test splits without re-fitting.
- Pairwise disjointness verification confirmed:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 4. Proposed Neural Network Architecture

The architecture (`MultimodalCrossAttentionClassifier` in `src/models/fusion.py`) comprises five modular stages:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODULAR ARCHITECTURE FLOW                                  │
│                                                                                        │
│  [mRNA (3217)]   ──▶ [Encoder MLP] ──▶ z_1 ──▶ [Token Subst + E_1] ──┐                 │
│  [miRNA (383)]   ──▶ [Encoder MLP] ──▶ z_2 ──▶ [Token Subst + E_2] ──┤                 │
│  [CNV (3105)]    ──▶ [Encoder MLP] ──▶ z_3 ──▶ [Token Subst + E_3] ──┼──▶ [Multi-Head] │
│  [Methy (3139)]  ──▶ [Encoder MLP] ──▶ z_4 ──▶ [Token Subst + E_4] ──┘    Self-Attn    │
│                                                                           (H=4, D=128) │
│                                                                                │       │
│  [Cancer Logits (4)] ◀── [Classifier Head] ◀── h_fused ◀── [Mask-Aware Pooling] ◀──┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Modality-Specific Encoders
Each modality $m \in \mathcal{M}$ is projected into a shared representation space $D = 128$ via a two-layer non-linear multilayer perceptron:
$$h_{m,1} = \text{Dropout}_{0.3}\left(\text{ReLU}\left(\text{BN}\left(W_{m,1} x_{m} + b_{m,1}\right)\right)\right)$$
$$z_{m} = \text{Dropout}_{0.3}\left(\text{ReLU}\left(\text{BN}\left(W_{m,2} h_{m,1} + b_{m,2}\right)\right)\right)$$
where $W_{m,1} \in \mathbb{R}^{256 \times K_m}$ and $W_{m,2} \in \mathbb{R}^{128 \times 256}$.

### 4.2 Learned Missing Tokens & Modality-Type Embeddings
To maintain fixed sequence length ($S = 4$) without numerical corruption, absent assays ($M_{i,m} = 0$) are replaced by a learnable default missing token $\theta_m \in \mathbb{R}^{128}$. A learnable modality-type embedding $E_m \in \mathbb{R}^{128}$ is added to preserve assay identity:
$$\tilde{z}_{i,m} = M_{i,m} z_{i,m} + (1.0 - M_{i,m}) \theta_m + E_m$$

### 4.3 Multi-Head Self-Attention over Modality Tokens
The token sequence $Z = [\tilde{z}_{i,1}, \tilde{z}_{i,2}, \tilde{z}_{i,3}, \tilde{z}_{i,4}] \in \mathbb{R}^{4 \times 128}$ undergoes bidirectional multi-head self-attention ($H = 4$ heads, key dimension $d_k = 32$):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \mathcal{M}_{\text{pad}}\right) V$$
where $\mathcal{M}_{\text{pad}} \in \{0, -\infty\}^{4 \times 4}$ is a key padding mask derived from $(M_i == 0)$, ensuring that absent modalities receive strictly zero attention weight. A residual connection, LayerNorm, feed-forward sublayer ($128 \to 256 \to 128$), and second LayerNorm yield contextualized tokens $Z'$.

### 4.4 Mask-Aware Mean Pooling
Contextualized tokens are aggregated into a patient representation vector $h_{\text{fused}} \in \mathbb{R}^{128}$ using mask-aware weighted mean pooling:
$$h_{\text{fused}} = \frac{\sum_{m=1}^4 M_{i,m} \cdot z'_{i,m}}{\max(1, \sum_{m=1}^4 M_{i,m})}$$
This guarantees that absent tokens do not distort the pooled representation.

### 4.5 Classification Head
The fused vector is projected into cancer class logits:
$$\hat{y} = W_2 \cdot \text{Dropout}_{0.3}\left(\text{ReLU}\left(W_1 h_{\text{fused}} + b_1\right)\right) + b_2$$
where $W_1 \in \mathbb{R}^{64 \times 128}$ and $W_2 \in \mathbb{R}^{4 \times 64}$.

---

## 5. Missing-Modality Simulation and Handling

### 5.1 15 Predefined Evaluation Conditions (C0–C14)

| Condition | Availability Vector $[M_1, M_2, M_3, M_4]$ | Present Modalities | Missing Description |
|:---:|:---:|:---|:---|
| **C0** | `[1, 1, 1, 1]` | mRNA, miRNA, CNV, Methy | Complete (Reference) |
| **C1** | `[0, 1, 1, 1]` | miRNA, CNV, Methy | mRNA missing |
| **C2** | `[1, 0, 1, 1]` | mRNA, CNV, Methy | miRNA missing |
| **C3** | `[1, 1, 0, 1]` | mRNA, miRNA, Methy | CNV missing |
| **C4** | `[1, 1, 1, 0]` | mRNA, miRNA, CNV | Methy missing |
| **C5** | `[0, 0, 1, 1]` | CNV, Methy | mRNA + miRNA missing |
| **C6** | `[0, 1, 0, 1]` | miRNA, Methy | mRNA + CNV missing |
| **C7** | `[0, 1, 1, 0]` | miRNA, CNV | mRNA + Methy missing |
| **C8** | `[1, 0, 0, 1]` | mRNA, Methy | miRNA + CNV missing |
| **C9** | `[1, 0, 1, 0]` | mRNA, CNV | miRNA + Methy missing |
| **C10** | `[1, 1, 0, 0]` | mRNA, miRNA | CNV + Methy missing |
| **C11** | `[0, 0, 0, 1]` | Methy | Only DNA Methylation present |
| **C12** | `[0, 0, 1, 0]` | CNV | **Only CNV present (Critical Stress Test)** |
| **C13** | `[0, 1, 0, 0]` | miRNA | Only miRNA present |
| **C14** | `[1, 0, 0, 0]` | mRNA | Only mRNA present |

### 5.2 Numerical Perturbation Invariance Proof
Formal unit tests (`test_masked_modality_numerical_invariance` in `tests/test_pipeline.py`) verify that altering numerical feature values of masked modalities ($M_{i,m} = 0$) with extreme random noise produces an absolute logit difference of **$\mathbf{0.0} < 10^{-6}$**.

---

## 6. Training Strategies and Experimental Design

### 6.1 Strategy S1 vs. Strategy S2
- **Strategy S1 (Standard Training):** Trained strictly on complete inputs ($p = 0.0$).
- **Strategy S2 (Modality Dropout Training):** In each training iteration, each modality is independently dropped with probability $p = 0.3$, subject to the constraint that $\sum M_m \ge 1$. This prevents co-adaptation and forces individual encoders to extract robust standalone representations.

### 6.2 Optimization Hyperparameters (Verified from Code)
- **Loss Function:** Multiclass Categorical Cross-Entropy (`nn.CrossEntropyLoss`).
- **Optimizer:** AdamW (learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$, gradient norm clipping $= 1.0$).
- **Learning Rate Scheduler:** Cosine Annealing (`CosineAnnealingLR`, $T_{\max} = 100, \eta_{\min} = 10^{-6}$).
- **Batch Size:** 64.
- **Maximum Epochs:** 100 with Early Stopping patience of 15 epochs monitored on validation loss ($\Delta_{\min} = 5 \times 10^{-4}$).
- **Multi-Seed Replication:** 5 fixed random seeds ($[42, 123, 456, 789, 1024]$) across all models.

---

## 7. Evaluation Framework and Paired Statistical Analysis

### 7.1 Statistical Methodology
Because S1 and S2 share identical test instances ($N=340$) and matching training seeds, unpaired two-sample tests are methodologically invalid. We execute:
1. **Paired Student's $t$-test ($df = 4$):** Evaluates whether the mean paired difference $\bar{D} = \frac{1}{5} \sum (S2_s - S1_s)$ differs significantly from zero.
2. **Paired Cohen's $d$:** Measures standardized effect size:
   $$d_{\text{paired}} = \frac{\bar{D}}{s_D}$$
3. **Wilcoxon Signed-Rank Test:** Non-parametric evaluation across the 5 paired differences.
4. **95% Confidence Interval for Mean Difference:**
   $$\text{CI}_{95\%} = \bar{D} \pm t_{0.975, 4} \cdot \frac{s_D}{\sqrt{5}}$$

### 7.2 Empirical Results on Condition C12 (Only CNV Present)
- Strategy S1 Macro-F1: $0.6561 \pm 0.0892$ (Seeds: $[0.7368, 0.5181, 0.6777, 0.7525, 0.5953]$)
- Strategy S2 Macro-F1: $0.7412 \pm 0.0476$ (Seeds: $[0.7413, 0.7897, 0.7897, 0.7413, 0.6441]$)
- Paired Mean Improvement ($\bar{D}$): $\mathbf{+0.0852}$ ($\mathbf{+8.52\%}$)
- Paired Standard Deviation ($s_D$): $0.1147$
- Paired Cohen's $d$: **$0.7425$** (Moderate-to-large empirical effect size)
- Paired $t$-statistic: $t = 1.6603, p = 0.1722$
- Wilcoxon signed-rank: $W = 2.0, p = 0.1875$
- 95% Confidence Interval: $[-0.0573, +0.2276]$

### 7.3 Methodological and Statistical Interpretation
- Strategy S2 improves over S1 on 4 out of 5 training seeds ($+0.0045, +0.2717, +0.1120, -0.0112, +0.0489$).
- Because $N=5$ seeds ($df=4$), the paired $t$-test yields $p = 0.1722$ and the confidence interval crosses zero.
- **Scientific Conclusion:** The observed $+8.52\%$ gain is an empirically substantial improvement, but is **not statistically significant at $\alpha = 0.05$** due to limited sample size ($N=5$).
- Standard deviations reflect **algorithmic training stochasticity**, not patient-population resample variance.

---

## 8. Implementation Timeline and Verification Status

| Stage | Milestone Description | Verification Source | Status |
|:---:|:---|:---|:---:|
| **1** | Cohort Selection & Audit | `scripts/verify_patient_leakage.py` | ✅ Complete (2,263 unique patients) |
| **2** | Preprocessing & Partitioning | `src/data/preprocessor.py`, `splits.csv` | ✅ Complete (70/15/15, train-only scalers) |
| **3** | Multimodal Architecture | `src/models/fusion.py` | ✅ Complete (Encoders, Attention, Pooling) |
| **4** | Missing-Modality Design | `src/data/dataset.py`, `test_pipeline.py` | ✅ Complete (15 conditions, invariance verified) |
| **5** | Multi-Seed Model Training | `artifacts/models/*.pt` | ✅ Complete (59 checkpoints preserved) |
| **6** | Statistical Hypothesis Testing | `scripts/run_statistical_tests.py` | ✅ Complete (Paired tests, exact CIs) |
| **7** | Presentation Visualizations | `reports/presentation_figures/` | ✅ Complete (24 high-res artifacts) |

### Explicit Identification of Pending Work
1. **External Hospital Cohort Validation:** Evaluated on TCGA Pan-Cancer only; external cohorts (e.g. CPTAC, ICGC) remain future work.
2. **Missing-Modality Baseline Imputation:** Baselines D5 (Early Fusion MLP) and D6 (LightGBM) were evaluated strictly on complete inputs (C0); missingness benchmarking with KNN/mean imputation remains future work.
3. **Statistical Power Scaling:** Evaluating with $N \ge 30$ seeds or 10-fold nested cross-validation is needed to establish formal non-parametric statistical significance.
