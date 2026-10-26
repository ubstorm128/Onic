# ONIC: Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities

## Comprehensive System & Architecture Documentation (`info.md`)

---

## Table of Contents
1. [Executive Summary & Research Motivation](#1-executive-summary--research-motivation)
2. [Dataset Specifications](#2-dataset-specifications)
   - [2.1 Data Origin & Pan-Cancer Cohort](#21-data-origin--pan-cancer-cohort)
   - [2.2 The 4 Biological Modalities](#22-the-4-biological-modalities)
   - [2.3 Preprocessing & Leak-Free Partitioning](#23-preprocessing--leak-free-partitioning)
3. [Model Architecture: Deep Dive](#3-model-architecture-deep-dive)
   - [3.1 Primary Proposed Model: MultimodalCrossAttentionClassifier](#31-primary-proposed-model-multimodalcrossattentionclassifier)
   - [3.2 Modality-Specific Encoders](#32-modality-specific-encoders)
   - [3.3 Dynamic Missingness Gate & Learned Default Tokens](#33-dynamic-missingness-gate--learned-default-tokens)
   - [3.4 Modality Type Embeddings](#34-modality-type-embeddings)
   - [3.5 Multi-Head Cross-Modal Transformer Core](#35-multi-head-cross-modal-transformer-core)
   - [3.6 Mask-Aware Token Pooling](#36-mask-aware-token-pooling)
   - [3.7 4-Way Classifier Head](#37-4-way-classifier-head)
4. [Mixture of Models & Comparative Baseline Ecosystem](#4-mixture-of-models--comparative-baseline-ecosystem)
   - [4.1 Role of Each Model in the System](#41-role-of-each-model-in-the-system)
   - [4.2 Baseline Models (D1 to D6)](#42-baseline-models-d1-to-d6)
   - [4.3 Primary Fusion Training Strategies (S1 vs S2 vs S3)](#43-primary-fusion-training-strategies-s1-vs-s2-vs-s3)
5. [Codebase Modules & Directory Layout](#5-codebase-modules--directory-layout)
   - [5.1 `project/src/models/` (Architectures)](#51-projectsrcmodels-architectures)
   - [5.2 `project/src/data/` (Pipeline & Preprocessing)](#52-projectsrcdata-pipeline--preprocessing)
   - [5.3 `project/src/training/` (Execution & Metrics)](#53-projectsrctraining-execution--metrics)
   - [5.4 `project/src/explainability/` (Attribution & Attention)](#54-projectsrcexplainability-attribution--attention)
   - [5.5 `project/scripts/` (Automated Audits & Experiments)](#55-projectscripts-automated-audits--experiments)
   - [5.6 `Code/` (Production Serving & Clinical Web Dashboard)](#56-code-production-serving--clinical-web-dashboard)
6. [How the System Works: End-to-End Operational Pipeline](#6-how-the-system-works-end-to-end-operational-pipeline)
   - [6.1 Offline Research & Training Flow](#61-offline-research--training-flow)
   - [6.2 Online Live Inference Flow](#62-online-live-inference-flow)
7. [Empirical Benchmarks & 15-Condition Missingness Matrix](#7-empirical-benchmarks--15-condition-missingness-matrix)
   - [7.1 All 15 Combinatorial Evaluation Conditions (C0–C14)](#71-all-15-combinatorial-evaluation-conditions-c0c14)
   - [7.2 Stress-Test Resilience (Condition C12: CNV Only)](#72-stress-test-resilience-condition-c12-cnv-only)
   - [7.3 ROC-AUC & Latent Space Geometry](#73-roc-auc--latent-space-geometry)
8. [Explainability & Biological Biomarker Concordance](#8-explainability--biological-biomarker-concordance)
   - [8.1 Autonomous Attention Weight Distribution](#81-autonomous-attention-weight-distribution)
   - [8.2 Cancer-Specific Driver Biomarkers (Input × Gradient)](#82-cancer-specific-driver-biomarkers-input--gradient)
9. [Ablation Study Insights](#9-ablation-study-insights)
10. [Academic & Project Information](#10-academic--project-information)

---

## 1. Executive Summary & Research Motivation

In modern computational oncology, high-throughput multi-omics assays—encompassing gene transcriptomics (mRNA), microRNA regulation (miRNA), somatic structural alterations (CNV), and epigenetic modifications (DNA Methylation)—provide complementary layers of biological information. While integrating these data streams significantly sharpens diagnostic accuracy, **clinical reality is plagued by the "missing modality problem"**:
- Tissue biopsies degrade (e.g., in archival formalin-fixed paraffin-embedded [FFPE] blocks).
- High sequencing costs prevent patients from receiving all four assays simultaneously.
- Clinical workflows are fragmented across different pathology labs.

Conventional multi-omics architectures (early-fusion concatenation MLPs, late-fusion voting, or simple zero-filling) suffer from **catastrophic performance collapse** when even a single assay stream is unobserved or noisy.

**ONIC** addresses this clinical challenge by introducing:
1. **Dedicated Modality Encoders** projecting disparate assay dimensions into a shared metric space ($\mathbb{R}^{128}$).
2. **Learned Default Missing Tokens** ($\mathbf{e}_{\text{default}}$) substituting absent modalities to preserve representation manifold geometry without zero-padding distortion.
3. **Multi-Head Cross-Modal Transformer Core** with key padding masking, enabling dynamic information routing across available modalities.
4. **Stochastic Modality Dropout Training** forcing the network to maintain high predictive power across all $2^4 - 1 = 15$ combinations of modality presence/absence.

---

## 2. Dataset Specifications

### 2.1 Data Origin & Pan-Cancer Cohort
- **Data Source:** Benchmark **MLOmics Pan-Cancer** dataset (Zenodo Archive `28729127.zip`, ~8.83 GB), curated from **The Cancer Genome Atlas (TCGA)**.
- **Harmonized Cohort:** Exactly **2,263 patient samples** across **4 major malignancies**.
- **Patient Independence:** Verified via `scripts/verify_patient_leakage.py` that 2,263 unique sample barcodes correspond to **2,263 unique patients** (zero multi-sample duplicate patients).

| Class Index | TCGA Label | Cancer Subtype | Full Malignancy Name | Patient Count | Cohort % |
|:---:|:---:|:---|:---|:---:|:---:|
| `0` | `2` | **BRCA** | Breast Invasive Carcinoma | 757 | 33.45% |
| `1` | `8` | **HNSC** | Head and Neck Squamous Cell Carcinoma | 506 | 22.36% |
| `2` | `13` | **LGG** | Brain Lower Grade Glioma | 506 | 22.36% |
| `3` | `27` | **THCA** | Thyroid Carcinoma | 494 | 21.83% |
| **Total** | — | — | — | **2,263** | **100.0%** |

### 2.2 The 4 Biological Modalities
The system integrates **9,844 total molecular features** across 4 assay layers:

| Modality Key | Biological Domain | Feature Count | Assay Platform & Technology | Normalization Standard | Raw Values Type |
|:---|:---|:---:|:---|:---|:---|
| **mRNA** | Transcriptomics | **3,217** genes | RNA-seq Gene Expression | `StandardScaler` (z-score) | $\log_2(\text{count} + 1)$ expression |
| **miRNA** | Epigenetic Regulation | **383** microRNAs | miRNA-seq Small RNA Sequencing | `StandardScaler` (z-score) | Reads per million / $\log_2$ |
| **CNV** | Genomic Alterations | **3,105** regions | Affymetrix SNP 6.0 / GISTIC2 | `StandardScaler` (z-score) | Segment focal copy number |
| **Methy** | DNA Methylation | **3,139** CpG probes | Illumina Infinium HumanMethylation450 | `MinMaxScaler` ($[0, 1]$ bounded) | Methylation $\beta$-values $\in [0, 1]$ |

### 2.3 Preprocessing & Leak-Free Partitioning
- **Partition Ratios:** Stratified 70% Train ($N=1,584$), 15% Validation ($N=339$), 15% Test ($N=340$).
- **Anti-Contamination Rule:** Normalization statistics (means, standard deviations, min/max bounds) were fitted **strictly on the 1,584 training samples** and saved as joblib artifacts (`preprocessor_mRNA.joblib`, `preprocessor_miRNA.joblib`, `preprocessor_CNV.joblib`, `preprocessor_Methy.joblib`).
- **Disjoint Partition Guarantee:** Verified pairwise zero overlap:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Quality Checks:** 26/26 automated audit checks passed (`reports/dataset_audit.json`), with 0 missing cells, 0 NaNs, and perfect positional sample-barcode alignment.

---

## 3. Model Architecture: Deep Dive

```
                             Patient Multi-Omics
                 ┌────────────────────┴────────────────────┐
                 ▼                    ▼                    ▼                    ▼
           mRNA (3,217-D)       miRNA (383-D)        CNV (3,105-D)        Methy (3,139-D)
                 │                    │                    │                    │
                 ▼                    ▼                    ▼                    ▼
          ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
          │ Encoder_mRNA │     │Encoder_miRNA │     │ Encoder_CNV  │     │Encoder_Methy │
          │ Linear(3217, │     │ Linear(383,  │     │ Linear(3105, │     │ Linear(3139, │
          │    256)->BN  │     │    256)->BN  │     │    256)->BN  │     │    256)->BN  │
          │ ->ReLU->Drop │     │ ->ReLU->Drop │     │ ->ReLU->Drop │     │ ->ReLU->Drop │
          │ Linear(256,  │     │ Linear(256,  │     │ Linear(256,  │     │ Linear(256,  │
          │    128)->BN  │     │    128)->BN  │     │    128)->BN  │     │    128)->BN  │
          └──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                 │                    │                    │                    │
                 ▼                    ▼                    ▼                    ▼
             z₁ ∈ ℝ¹²⁸            z₂ ∈ ℝ¹²⁸            z₃ ∈ ℝ¹²⁸            z₄ ∈ ℝ¹²⁸
                 │                    │                    │                    │
                 └────────────────────┼────────────────────┼────────────────────┘
                                      ▼
                      ┌─────────────────────────────────┐
                      │    Dynamic Missingness Gate     │
                      │ z̃ᵢ = mᵢ·zᵢ + (1 - mᵢ)·e_default⁽ⁱ⁾ │
                      └───────────────┬─────────────────┘
                                      │ + Modality Type Embeddings (E_type ∈ ℝ¹ˣ⁴ˣ¹²⁸)
                                      ▼
                      ┌─────────────────────────────────┐
                      │ Multi-Head Self-Attention Core  │
                      │ 4 Heads, d_k=32, embed_dim=128  │
                      │ Key Padding Mask (mask == 0)    │
                      │ Residual + LayerNorm1           │
                      │ FeedForward (128 -> 256 -> 128) │
                      │ Residual + LayerNorm2           │
                      └───────────────┬─────────────────┘
                                      ▼
                      ┌─────────────────────────────────┐
                      │    Mask-Aware Mean Pooling      │
                      │  h_fused = (∑ mᵢ·Z'ᵢ) / (∑ mᵢ)  │
                      └───────────────┬─────────────────┘
                                      ▼
                      ┌─────────────────────────────────┐
                      │       4-Way Classifier Head     │
                      │ Linear(128, 64) -> ReLU -> Drop │
                      │        Linear(64, 4)            │
                      └───────────────┬─────────────────┘
                                      ▼
                      ┌───────┬───────┴───────┬───────┐
                      ▼       ▼               ▼       ▼
                    BRCA    HNSC             LGG    THCA
```

### 3.1 Primary Proposed Model: `MultimodalCrossAttentionClassifier`
Implemented in [`project/src/models/fusion.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/models/fusion.py), this is the central neural network of ONIC. It processes raw multi-omics feature dictionaries alongside a binary observation mask $m \in \{0, 1\}^4$.

### 3.2 Modality-Specific Encoders
Because each assay has fundamentally different dimensionalities ($383$ to $3,217$) and scale characteristics, a single monolithic network causes high-dimensional assays to overwhelm smaller ones. 
- Each modality has a dedicated 2-layer MLP (`ModalityEncoder`):
  $$\mathbf{h}_{m,1} = \text{Dropout}_{0.3}\left(\text{ReLU}\left(\text{BatchNorm1d}\left(\mathbf{W}_{m,1} \mathbf{x}_m\right)\right)\right), \quad \mathbf{W}_{m,1} \in \mathbb{R}^{256 \times K_m}$$
  $$\mathbf{z}_m = \text{Dropout}_{0.3}\left(\text{ReLU}\left(\text{BatchNorm1d}\left(\mathbf{W}_{m,2} \mathbf{h}_{m,1}\right)\right)\right), \quad \mathbf{W}_{m,2} \in \mathbb{R}^{128 \times 256}$$
- Maps every modality to a common metric latent dimension $D = 128$.

### 3.3 Dynamic Missingness Gate & Learned Default Tokens
When a modality is missing ($m_i = 0$):
- Naive zero-padding pulls representations toward origin artifacts, distorting downstream attention.
- ONIC uses a learnable parameter tensor $\mathbf{\Theta}_{\text{default}} \in \mathbb{R}^{4 \times 128}$ (`default_missing_tokens`):
  $$\tilde{\mathbf{z}}_i = m_i \mathbf{z}_i + (1 - m_i) \mathbf{e}_{\text{default}}^{(i)}$$
- The learned default vector represents the mathematical concept of *"unobserved modality prior"* and is optimized via backpropagation.

### 3.4 Modality Type Embeddings
To provide identity context within the transformer sequence (similar to positional embeddings in NLP):
- A learned parameter tensor $\mathbf{E}_{\text{type}} \in \mathbb{R}^{1 \times 4 \times 128}$ is added to the token sequence:
  $$\mathbf{Z}_0 = [\tilde{\mathbf{z}}_1, \tilde{\mathbf{z}}_2, \tilde{\mathbf{z}}_3, \tilde{\mathbf{z}}_4]^\top + \mathbf{E}_{\text{type}}$$

### 3.5 Multi-Head Cross-Modal Transformer Core
Cross-modal interaction is governed by `nn.MultiheadAttention`:
- **Attention Configuration:** `embed_dim = 128`, `num_heads = 4` ($d_k = 32$), `dropout = 0.1`, `batch_first = True`.
- **Key Padding Mask:** Missing modalities are flagged as `True` in `key_padding_mask` ($m_i = 0$), strictly excluding absent tokens from attention weighting.
- **Numerical Safeguard:** If all modalities are missing for a sample, the mask unmasks all tokens to prevent `NaN` values in the softmax function.
- **Transformer Block:**
  $$\mathbf{Z}_{\text{attn}} = \text{LayerNorm}\left(\mathbf{Z}_0 + \text{MultiheadAttention}(\mathbf{Z}_0, \mathbf{Z}_0, \mathbf{Z}_0)\right)$$
  $$\mathbf{Z}_{\text{final}} = \text{LayerNorm}\left(\mathbf{Z}_{\text{attn}} + \text{FFN}(\mathbf{Z}_{\text{attn}})\right)$$
  where $\text{FFN}(\mathbf{u}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{u})$ with hidden dimension $256$.

### 3.6 Mask-Aware Token Pooling
Rather than pooling over all 4 tokens (which would incorporate missing/default representations), the aggregation is mask-weighted:
$$\mathbf{h}_{\text{fused}} = \frac{\sum_{i=1}^4 m_i \mathbf{Z}_{\text{final}, i}}{\max\left(1.0, \sum_{i=1}^4 m_i\right)} \in \mathbb{R}^{128}$$

### 3.7 4-Way Classifier Head
Projects the patient vector to 4 cancer class logits:
$$\mathbf{h}_{\text{cls}} = \text{Dropout}_{0.3}\left(\text{ReLU}\left(\mathbf{W}_{\text{c1}} \mathbf{h}_{\text{fused}} + \mathbf{b}_1\right)\right), \quad \mathbf{W}_{\text{c1}} \in \mathbb{R}^{64 \times 128}$$
$$\text{logits} = \mathbf{W}_{\text{c2}} \mathbf{h}_{\text{cls}} + \mathbf{b}_2, \quad \mathbf{W}_{\text{c2}} \in \mathbb{R}^{4 \times 64}$$
$$\mathbf{p} = \text{Softmax}(\text{logits}) \in [0, 1]^4$$

---

## 4. Mixture of Models & Comparative Baseline Ecosystem

The repository contains both the **proposed primary architecture** and a suite of **baseline models** used to benchmark and mathematically validate every design choice:

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    MODEL ECOSYSTEM                     │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
               ┌───────────────────────────────────────────┴──────────────────────────────────────────┐
               ▼                                                                                      ▼
 ┌───────────────────────────┐                                                          ┌───────────────────────────┐
 │   Baseline Suite (D1-D6)  │                                                          │   Primary Architecture    │
 ├───────────────────────────┤                                                          ├───────────────────────────┤
 │ D1: mRNA Single MLP       │                                                          │ Multimodal Cross-Attention│
 │ D2: miRNA Single MLP      │                                                          │ Classifier                │
 │ D3: CNV Single MLP        │                                                          │ (CrossAttentionClassifier)│
 │ D4: Methy Single MLP      │                                                          └─────────────┬─────────────┘
 │ D5: Early Fusion MLP      │                                                                        │
 │ D6: LightGBM GBDT         │                                           ┌────────────────────────────┴───────────────────────────┐
 └───────────────────────────┘                                           ▼                                                        ▼
                                                           ┌───────────────────────────┐                            ┌───────────────────────────┐
                                                           │ Strategy S1 (Complete)    │                            │ Strategy S2 (Dropout)     │
                                                           │ Trained on all 4 omics    │                            │ Trained with p=0.3 dropout│
                                                           │ Fails under missingness   │                            │ Resilient across all 15   │
                                                           │ (Reference standard)      │                            │ missingness conditions    │
                                                           └───────────────────────────┘                            └───────────────────────────┘
```

### 4.1 Role of Each Model in the System

| Model Identifier | Architecture / Algorithm | Where It Operates | Purpose in the System | Test Macro-F1 (Mean ± Std) |
|:---:|:---|:---|:---|:---:|
| **D1** | `SingleModalityMLP` | `project/src/models/baselines.py` | Benchmark: Evaluates predictive power of **mRNA alone** (3,217-D). | $0.9961 \pm 0.0015$ |
| **D2** | `SingleModalityMLP` | `project/src/models/baselines.py` | Benchmark: Evaluates predictive power of **miRNA alone** (383-D). | $0.9967 \pm 0.0012$ |
| **D3** | `SingleModalityMLP` | `project/src/models/baselines.py` | Benchmark: Evaluates predictive power of **CNV alone** (3,105-D). Shows structural noise. | $0.8532 \pm 0.0114$ |
| **D4** | `SingleModalityMLP` | `project/src/models/baselines.py` | Benchmark: Evaluates predictive power of **DNA Methylation alone** (3,139-D). | $0.9945 \pm 0.0000$ |
| **D5** | `EarlyFusionMLP` | `project/src/models/baselines.py` | Benchmark: Naive concatenation of all 9,844 features. C0 reference. | $0.9961 \pm 0.0015$ |
| **D6** | `LightGBMBaseline` | `project/src/models/baselines.py` | Benchmark: Gradient-boosted decision trees. Tests tabular baseline. | $\mathbf{0.9973 \pm 0.0000}$ |
| **E1 (S1)** | `MultimodalCrossAttentionClassifier` | `project/src/models/fusion.py` | Primary model trained **without dropout** ($p=0.0$). Reference fusion. | $0.9956 \pm 0.0015$ |
| **E2 (S2)** | `MultimodalCrossAttentionClassifier` | `project/src/models/fusion.py` & `Code/api_server.py` | **Proposed resilient model** trained with **Modality Dropout** ($p=0.3$). Production default. | $0.9950 \pm 0.0012$ (Complete)<br>$\mathbf{0.7412 \pm 0.0476}$ (**CNV-only C12**) |

### 4.2 Baseline Models (D1 to D6)
1. **Single-Modality MLPs (D1–D4):**
   - Architecture: $\text{Input}(K) \to \text{Linear}(K, 256) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(256, 128) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128, 4)$.
   - Key Finding: mRNA, miRNA, and Methylation alone achieve $>0.994$ Macro-F1. However, CNV achieves only $0.8532$, proving that genomic copy number variations suffer from high intra-cancer heterogeneity and noise.
2. **Early Fusion Concatenation MLP (D5):**
   - Concatenates all modalities into a 9,844-dimensional vector.
   - Architecture: $\text{Linear}(9844, 512) \to \text{BN} \to \text{ReLU} \to \text{Linear}(512, 256) \to \dots \to \text{Linear}(128, 4)$.
   - Flaw: When any modality is missing, zeroing out features distorts activations and collapses classification.
3. **LightGBM Tabular GBDT (D6):**
   - 200 trees, learning rate 0.05, 31 leaves.
   - Strongest baseline on complete data ($0.9973$), but lacks neural attention gating and cannot dynamically route around missing assays without extensive tabular imputation.

### 4.3 Primary Fusion Training Strategies (S1 vs S2 vs S3)
- **Strategy S1 (Complete Reference):** Modality dropout probability $p = 0.0$. The network only ever sees complete 4-omics profiles during training. At test time under missing conditions, it relies on learned default vectors, but experiences performance drops when noisy assays (like CNV) are isolated.
- **Strategy S2 (Modality Dropout - Proposed):** Modality dropout probability $p = 0.3$. During training, modalities are randomly masked with $30\%$ probability per sample (ensuring at least 1 remains active). This forces encoders and attention heads to learn robust independent and partial cross-modal representations.
- **Strategy S3 (Curriculum Modality Dropout):** Linearly warms up dropout from $p=0.0$ to $p=0.4$ over 10 epochs.

---

## 5. Codebase Modules & Directory Layout

```
Onic/
├── assets/                                 # Brand assets, SVGs, workflow diagrams
│   ├── banner.svg                          # High-res project banner
│   └── model_workflow.svg                  # Animated SVG of cross-attention pipeline
├── Code/                                   # Full-Stack Serving & Web Application
│   ├── index.html                          # Clinical Prediction Studio UI
│   ├── style.css                           # Modern CSS design system
│   ├── script.js                           # Frontend client & real-time telemetry
│   ├── api_server.py                       # FastAPI REST API backend
│   ├── inference_service.py                # PyTorch runtime engine & input parsing
│   ├── templates/                          # Modality CSV upload templates
│   └── sample_inputs/                      # Held-out patient verification samples
├── project/                                # Machine Learning Research Engine
│   ├── configs/                            # YAML parameter specifications
│   │   ├── data.yaml                       # Cohort, paths, and normalization rules
│   │   ├── model.yaml                      # Layer dimensions, heads, and dropout
│   │   └── experiment.yaml                 # Seeds, batch sizes, and loss metrics
│   ├── docs/                               # 12 Research & Protocol Specifications
│   ├── src/                                # Core Python Source Code
│   │   ├── data/                           # Dataset loaders, splitting, scaling
│   │   ├── models/                         # Fusion & Baseline PyTorch models
│   │   ├── training/                       # Seeded trainer, losses, metrics
│   │   └── explainability/                 # Feature & modality attribution
│   ├── scripts/                            # Reproducibility & audit runners
│   ├── reports/                            # Publication figures, tables, logs
│   ├── artifacts/                          # Trained checkpoints (.pt) & metrics
│   └── tests/                              # Automated unit & integration test suite
├── TCGA_Dataset/                           # Raw TCGA multi-omics matrices
└── info.md                                 # Complete system information document
```

### 5.1 `project/src/models/` (Architectures)
- [`fusion.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/models/fusion.py): Defines `ModalityEncoder` and `MultimodalCrossAttentionClassifier`. Implements learned default missingness tokens, modality type embeddings, PyTorch MultiheadAttention, and mask-aware mean pooling.
- [`baselines.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/models/baselines.py): Defines `SingleModalityMLP`, `EarlyFusionMLP`, and `LightGBMBaseline`.

### 5.2 `project/src/data/` (Pipeline & Preprocessing)
- [`dataset.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/data/dataset.py): `MultimodalCancerDataset` PyTorch `Dataset` wrapper. Handles on-the-fly random modality dropout during training and deterministic mask generation for test conditions C0–C14.
- [`preprocessor.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/data/preprocessor.py): Computes leak-free scaling (`StandardScaler` for mRNA, miRNA, CNV; `MinMaxScaler` for Methylation) strictly on the training partition. Serializes to `.joblib`.
- [`splitting.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/data/splitting.py): Executes stratified 70/15/15 splits preserving class distributions without patient overlap.

### 5.3 `project/src/training/` (Execution & Metrics)
- [`trainer.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/training/trainer.py): Manages multi-seed training loops with early stopping (patience = 15 epochs), cosine learning rate scheduler, gradient clipping (norm = 1.0), and checkpoint saving.
- [`losses.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/training/losses.py): Cross-entropy loss with optional class weighting.
- [`metrics.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/training/metrics.py): Evaluates Macro-F1, Micro-F1, Weighted-F1, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Multiclass ROC-AUC, and Confusion Matrices.

### 5.4 `project/src/explainability/` (Attribution & Attention)
- [`modality_importance.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/explainability/modality_importance.py): Computes inter-modality cross-attention matrices, permutation importance, and ablation drops.
- [`feature_importance.py`](file:///d:/PLACE/BCA/MajorProject/Onic/project/src/explainability/feature_importance.py): Computes gradient-based feature sensitivity ($\text{Input} \times |\nabla_x|$) to identify top driving biomarkers per cancer type.

### 5.5 `project/scripts/` (Automated Audits & Experiments)
- `audit_dataset.py`: 26-point automated data audit.
- `prepare_dataset.py`: Converts raw CSV matrices into processed `.npy` arrays.
- `run_baselines.py`: Trains and evaluates all 6 baselines (D1–D6) across 5 seeds.
- `run_fusion.py`: Trains proposed cross-attention model under S1 and S2 strategies across 5 seeds.
- `run_missing_modality_experiments.py`: Evaluates all 15 conditions (C0–C14).
- `run_ablations.py`: Runs the 4 ablation studies (tokens, dropout prob, latent dim, heads).
- `run_statistical_tests.py`: Computes paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d$, and 95% CIs.
- `generate_presentation_figures.py`: Generates 24 publication-ready figures (300 DPI).

### 5.6 `Code/` (Production Serving & Clinical Web Dashboard)
- [`inference_service.py`](file:///d:/PLACE/BCA/MajorProject/Onic/Code/inference_service.py): `MultimodalInferenceEngine` singleton. Loads PyTorch checkpoints, fitted scalers, test patient data, handles CSV upload parsing, feature name matching, and executes inference in `torch.inference_mode()`.
- [`api_server.py`](file:///d:/PLACE/BCA/MajorProject/Onic/Code/api_server.py): FastAPI REST service with endpoints:
  - `POST /api/predict`: JSON prediction with patient index or manual feature vectors.
  - `POST /api/predict/upload`: Multipart CSV/TSV upload for any combination of modalities.
  - `GET /api/template/{modality}`: Downloadable pre-formatted CSV template.
  - `GET /api/patients`: Metadata for held-out TCGA test patients.
  - `GET /api/health`: Health status, loaded checkpoints, and preprocessor verification.
- [`index.html`](file:///d:/PLACE/BCA/MajorProject/Onic/Code/index.html), [`style.css`](file:///d:/PLACE/BCA/MajorProject/Onic/Code/style.css), [`script.js`](file:///d:/PLACE/BCA/MajorProject/Onic/Code/script.js): Interactive Clinical Prediction Studio featuring:
  - Dynamic modality toggle switches (15 conditions).
  - Pre-loaded patient presets (BRCA #101, HNSC #142, LGG #188, THCA #215).
  - Diagnostic prediction card with real-time probability distribution gauges.
  - Biomarker attribution list and modality contribution radar.
  - Offline fallback engine automatically upgrading to live model upon API connection.

---

## 6. How the System Works: End-to-End Operational Pipeline

### 6.1 Offline Research & Training Flow

```
[Raw TCGA Pan-Cancer CSVs] (mRNA: 3217, miRNA: 383, CNV: 3105, Methy: 3139)
                          │
                          ▼
            [26-Check Automated Audit] ──▶ (Passed: 0 NaNs, 2263 unique patients)
                          │
                          ▼
          [Stratified 70/15/15 Split] ──▶ Train (1584), Val (339), Test (340)
                          │
                          ▼
         [Anti-Leakage Scaler Fitting] ──▶ StandardScaler (mRNA, miRNA, CNV)
                          │                MinMaxScaler (Methy)
                          ▼
       [MultimodalCrossAttention Training]
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
   Strategy S1: p=0.0            Strategy S2: p=0.3
   (Complete Data Only)          (Modality Dropout Regularization)
           │                             │
           └──────────────┬──────────────┘
                          ▼
     [15-Condition Benchmark Across 5 Seeds] ──▶ C0 to C14 Evaluated
                          │
                          ▼
        [Statistical Tests & Explainability] ──▶ Attention Routing & Biomarkers
```

### 6.2 Online Live Inference Flow

```
User Action: Uploads CSVs / Selects Patient Case / Toggles Active Modalities
                                    │
                                    ▼
       [FastAPI Web Server] (`Code/api_server.py: POST /api/predict`)
                                    │
                                    ▼
      [Inference Engine] (`Code/inference_service.py`)
       1. Validates file headers & feature dimensions
       2. Applies training-fitted StandardScaler / MinMaxScaler
       3. Assembles binary modality presence mask m = [m₁, m₂, m₃, m₄]
       4. Checks: If m == [0,0,0,0] ──▶ Rejects with HTTP 400 Bad Request
                                    │
                                    ▼
      [PyTorch MultimodalCrossAttentionClassifier]
       1. Encoders project present modalities to ℝ¹²⁸
       2. Missing modalities substituted with learned default tokens e_default
       3. Modality type embeddings added
       4. Multihead Attention routes information (key_padding_mask on missing)
       5. Mask-aware pooling aggregates observed tokens into h_fused
       6. Linear classifier outputs logits ──▶ Softmax probabilities
                                    │
                                    ▼
      [Explainability & Telemetry Extraction]
       1. Attention weight matrix extracted
       2. Modality contribution percentages calculated
       3. Top driving cancer biomarkers retrieved
       4. Sub-millisecond latency recorded
                                    │
                                    ▼
      [JSON Response] ──▶ Returned to Clinical UI with Live Diagnostic Visualizations
```

---

## 7. Empirical Benchmarks & 15-Condition Missingness Matrix

### 7.1 All 15 Combinatorial Evaluation Conditions (C0–C14)
Evaluated across **2,263 patient samples** across **5 random seeds** ([42, 123, 456, 789, 1024]) on the unseen test partition ($N=340$):

| Condition ID | Description | Active Modalities | Missing Modalities | S1 (Complete Reference) Macro-F1 | S2 (Modality Dropout) Macro-F1 | Empirical Gain (Δ) | Cohen's $d$ Effect Size |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | **Complete Multi-Omic** | **mRNA, miRNA, CNV, Methy** | *None* | **0.9956 ± 0.0015** | **0.9950 ± 0.0012** | -0.0006 | -0.24 |
| **C1** | Missing mRNA | miRNA, CNV, Methy | mRNA | 0.9939 ± 0.0012 | **0.9950 ± 0.0012** | **+0.0011** | +0.45 |
| **C2** | Missing miRNA | mRNA, CNV, Methy | miRNA | 0.9945 ± 0.0000 | 0.9945 ± 0.0000 | 0.0000 | 0.00 |
| **C3** | Missing CNV | mRNA, miRNA, Methy | CNV | 0.9956 ± 0.0015 | **0.9967 ± 0.0012** | **+0.0011** | +0.45 |
| **C4** | Missing Methy | mRNA, miRNA, CNV | Methy | 0.9956 ± 0.0015 | **0.9967 ± 0.0012** | **+0.0011** | +0.45 |
| **C5** | Missing mRNA + miRNA | CNV, Methy | mRNA, miRNA | 0.9934 ± 0.0025 | **0.9938 ± 0.0015** | **+0.0004** | +0.14 |
| **C6** | Missing mRNA + CNV | miRNA, Methy | mRNA, CNV | 0.9956 ± 0.0015 | 0.9956 ± 0.0015 | 0.0000 | 0.00 |
| **C7** | Missing mRNA + Methy | miRNA, CNV | mRNA, Methy | 0.9928 ± 0.0015 | **0.9939 ± 0.0030** | **+0.0011** | +0.26 |
| **C8** | Missing miRNA + CNV | mRNA, Methy | miRNA, CNV | 0.9950 ± 0.0012 | 0.9950 ± 0.0012 | 0.0000 | 0.00 |
| **C9** | Missing miRNA + Methy | mRNA, CNV | miRNA, Methy | 0.9950 ± 0.0012 | 0.9950 ± 0.0012 | 0.0000 | 0.00 |
| **C10** | Missing CNV + Methy | mRNA, miRNA | CNV, Methy | 0.9973 ± 0.0000 | 0.9973 ± 0.0000 | 0.0000 | 0.00 |
| **C11** | Methy Only (75% Missing) | Methy | mRNA, miRNA, CNV | 0.9923 ± 0.0050 | **0.9933 ± 0.0028** | **+0.0010** | +0.16 |
| **C12** | **CNV Only (Stress Test)** | **CNV** | **mRNA, miRNA, Methy** | **0.6561 ± 0.0892** | **0.7412 ± 0.0476** | **+0.0852 (+8.52%)** | **+0.74 (Large)** |
| **C13** | miRNA Only (75% Missing) | miRNA | mRNA, CNV, Methy | 0.9915 ± 0.0034 | **0.9945 ± 0.0033** | **+0.0030** | +0.68 |
| **C14** | mRNA Only (75% Missing) | mRNA | miRNA, CNV, Methy | 0.9956 ± 0.0015 | **0.9973 ± 0.0000** | **+0.0017** | **+1.10** |

### 7.2 Stress-Test Resilience (Condition C12: CNV Only)
- In Condition C12, 75% of diagnostic input channels are missing, leaving only genomic copy number alterations.
- Strategy S1 drops to $0.6561 \pm 0.0892$.
- Strategy S2 maintains $\mathbf{0.7412 \pm 0.0476}$, achieving an **$+8.52\%$ absolute F1 improvement** (with ablations scaling up to $+13.36\%$ under $p=0.5$).
- Modality dropout training prevents catastrophic breakdown by forcing the CNV encoder to extract maximal discriminative features rather than relying on transcriptomic shortcuts.

### 7.3 ROC-AUC & Latent Space Geometry
- **Multiclass ROC-AUC:** Macro-AUC reaches **0.9996**, with individual AUCs:
  - BRCA: $1.0000$
  - HNSC: $0.9974$
  - LGG: $1.0000$
  - THCA: $1.0000$
- **t-SNE Latent Space Projections:** 2D projections of the 128-dimensional fused embeddings demonstrate tight, distinct clusters for all 4 tumor types with wide inter-class margins and minimal boundary overlap.

---

## 8. Explainability & Biological Biomarker Concordance

### 8.1 Autonomous Attention Weight Distribution
Extracting learned self-attention weights across complete quad-modal inputs reveals how the transformer routes information:

| Query Modality | Attention to mRNA | Attention to miRNA | Attention to CNV | Attention to Methy |
|:---|:---:|:---:|:---:|:---:|
| **mRNA** | 34.01% | 26.10% | **5.69%** | 34.21% |
| **miRNA** | 34.40% | 26.49% | **5.39%** | 33.72% |
| **CNV** | 30.86% | 28.42% | **10.66%** | 30.06% |
| **Methy** | 34.49% | 25.27% | **5.36%** | 34.88% |

**Key Insight:** The cross-attention mechanism autonomously suppresses the noisy CNV channel (allocating only $5.7\%$ attention), while routing over $68.2\%$ of attention through high-confidence transcriptomic (mRNA: $34.0\%$) and epigenetic (DNA Methylation: $34.2\%$) features.

### 8.2 Cancer-Specific Driver Biomarkers (Input × Gradient)
Using gradient-weighted feature attribution ($\text{Input} \times |\nabla_x|$), the system extracts top driving molecular markers, matching canonical clinical literature:

| Target Cancer Class | Top mRNA Gene | Top microRNA | Top CNV Alteration | Top Methylation Probe | Clinical Biological Role |
|:---|:---|:---|:---|:---|:---|
| **BRCA** (Breast) | `SFRP1` | `hsa-miR-10b-3p` | `TTI2` | `cg10818284` | *SFRP1* is a canonical Wnt antagonist silenced in invasive breast cancers; *miR-10b* governs metastatic dissemination. |
| **HNSC** (Head & Neck) | `MIA` / `ERN2` | `hsa-miR-6720-3p` | `ANO1` | `cg27342919` | *ANO1* (Anoctamin-1) amplification at 11q13 is a hallmark biomarker of oral and head/neck squamous cell carcinoma. |
| **LGG** (Glioma) | `CNTNAP4` | `hsa-miR-200b-5p` | `PDGFRA` | `cg24882673` | *PDGFRA* is a classic receptor tyrosine kinase oncogene driving lower grade glioma tumorigenesis. |
| **THCA** (Thyroid) | `SLC26A4` | `hsa-let-7i-3p` | `CHRM3` / `DPP4` | `cg01303372` | *DPP4* (CD26) is a validated clinical diagnostic immunohistochemical marker for papillary thyroid carcinoma. |

---

## 9. Ablation Study Insights

A systematic 4-tier ablation suite ($N=24$ multi-seed runs) evaluated every architectural component:

1. **Learned Default Token vs Zero-Vector Masking:**
   - Zero-masking creates gradient volatility, resulting in a **35.2% wider variance** ($\sigma = 0.1263$ vs $0.0934$). Learned default tokens regularize embedding trajectories.
2. **Dropout Probability ($p \in \{0.1, 0.3, 0.5\}$):**
   - Robustness scales monotonically with dropout rate. Setting $p=0.5$ boosts CNV-only performance to $\mathbf{0.7897 \pm 0.0107}$ (an absolute gain of $+13.36\%$ over S1) without harming complete data performance ($0.9945$).
3. **Latent Representation Dimension ($D \in \{64, 128, 256\}$):**
   - Compact latent spaces ($D=64$ and $D=128$) constrain the representation manifold better than overparameterized $D=256$, preventing inter-modality co-adaptation.
4. **Attention Heads ($H \in \{1, 2, 4\}$):**
   - Single-head attention ($H=1$) suffers catastrophic collapse on CNV-only inputs down to $0.5907 \pm 0.1094$. Multi-head attention ($H=4$) is required to decouple orthogonal biological subspaces.

---

## 10. Academic & Project Information

- **Degree:** Bachelor of Computer Applications (BCA) Major Project
- **Faculty:** Faculty of Computer Technology
- **Institution:** [Assam down town University, Guwahati](https://adtu.in)
- **Academic Supervisor:** Mr. Wanraplang Nongbri
- **Research & Development Team:**
  - **Aniket Saha** — *Data Engineering & Deep Learning Architecture*
  - **Udit Bora** — *Lead Developer & Interface Engineering*
  - **Kritartha Barman** — *Evaluation Protocols & Leak-Free Pipeline*

### Running the System
```bash
# 1. Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Launch FastAPI backend & Clinical UI
python Code/api_server.py

# 3. Access in browser:
# UI: http://127.0.0.1:8000/
# API Docs: http://127.0.0.1:8000/docs
```
