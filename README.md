<p align="center">
  <img src="assets/banner.svg" alt="ONIC: Multimodal AI Cancer Classification Banner" width="100%"/>
</p>

<p align="center">
  <a href="#animated-architecture-in-action"><img src="https://img.shields.io/badge/Architecture-Cross--Attention%20Transformer-1e293b?style=for-the-badge&logo=pytorch&logoColor=white" alt="Architecture"/></a>
  <a href="#empirical-benchmarks-15-combinatorial-states"><img src="https://img.shields.io/badge/Peak%20Macro--F1-99.56%25-047857?style=for-the-badge" alt="Macro F1"/></a>
  <a href="#empirical-benchmarks-15-combinatorial-states"><img src="https://img.shields.io/badge/Macro--AUC-0.9996-1e40af?style=for-the-badge" alt="Macro AUC"/></a>
  <a href="#clinical-web-simulator--telemetry"><img src="https://img.shields.io/badge/Clinical%20UI-FastAPI%20%2B%20Vanilla%20JS-334155?style=for-the-badge&logo=fastapi&logoColor=white" alt="Web UI"/></a>
  <a href="#academic-credits--team"><img src="https://img.shields.io/badge/Institution-Assam%20down%20town%20University-4338ca?style=for-the-badge" alt="Institution"/></a>
</p>

<p align="center">
  <b>A research-grade multimodal deep learning framework designed to maintain clinical diagnostic accuracy across 4 major cancer malignancies even under severe diagnostic record missingness.</b>
</p>

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Animated Architecture in Action](#animated-architecture-in-action)
- [Architectural Novelty](#architectural-novelty)
- [Clinical Web Simulator & Telemetry](#clinical-web-simulator--telemetry)
- [Empirical Benchmarks (15 Combinatorial States)](#empirical-benchmarks-15-combinatorial-states)
- [Latent Space & Diagnostic Visualizations](#latent-space--diagnostic-visualizations)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Academic Credits & Team](#academic-credits--team)

---

## Research Motivation

In computational oncology, high-throughput molecular profiling (such as transcriptomics, genomics, and epigenetics) provides rich, complementary signals for cancer classification. However, **in real-world clinical pathology, complete multi-omic profiles are rarely available** for every patient due to tissue degradation, prohibitive sequencing costs, or fragmented diagnostic workflows.

Standard machine learning models (early fusion concatenations, late fusion voting, or naive imputation) suffer from **catastrophic performance collapse** when even a single assay stream is unobserved. 

**ONIC** solves this clinical barrier through:
1. **Dedicated Modality Encoders** mapping disparate biological dimensions into a unified metric space ($\mathbb{R}^{128}$).
2. **Learned Default Missingness Imputation** that avoids zero-padding artifacts and maintains representation manifold geometry.
3. **Multi-Head Cross-Modal Self-Attention** dynamically routing information between observed streams.
4. **Modality Dropout Training**, training the network across random missing subsets so it degrades gracefully rather than failing catastrophically.

---

## Animated Architecture in Action

The following vector animation demonstrates how patient omic profiles flow through dedicated MLPs, pass through the **Resilience Gate** (substituting missing assays with learned latent default vectors $\mathbf{e}_{\text{default}}$), cross-attend in the transformer core, and produce real-time subtype diagnoses:

<p align="center">
  <img src="assets/model_workflow.svg" alt="Animated Model Workflow" width="100%"/>
</p>

> [!TIP]
> Notice how the **CNV stream** is dynamically flagged as missing ($m_3 = 0$) in the simulator above: instead of zero-filling or dropping the patient record, the framework seamlessly injects a learned default token $\mathbf{e}_{\text{default}}^{(3)}$, allowing the remaining mRNA, miRNA, and Methylation streams to maintain cohesive cross-modal attention!

---

## Architectural Novelty

```
                          ┌────────────────────────┐
                          │ Patient Multi-Omics    │
                          │ (4 High-Dim Modalities)│
                          └───────────┬────────────┘
                                      │
         ┌───────────────┬────────────┴───┬───────────────┐
         ▼               ▼                ▼               ▼
   mRNA (3,217-D)  miRNA (383-D)    CNV (3,105-D)   DNA Meth (3,139-D)
         │               │                │               │
         ▼               ▼                ▼               ▼
   [Encoder MLP]   [Encoder MLP]    [Encoder MLP]   [Encoder MLP]
         │               │                │               │
         ▼               ▼                ▼               ▼
     z₁ ∈ ℝ¹²⁸       z₂ ∈ ℝ¹²⁸        z₃ ∈ ℝ¹²⁸       z₄ ∈ ℝ¹²⁸
         │               │                │               │
         └───────────────┼────────────────┼───────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │ Dynamic Missingness Gate          │
        │ zᵢ ← mᵢ·zᵢ + (1 - mᵢ)·e_default⁽ⁱ⁾ │
        └─────────────────┬─────────────────┘
                          │ + Modality Type Embeddings
                          ▼
        ┌───────────────────────────────────┐
        │ Multi-Head Self-Attention Core    │
        │ 4 Attention Heads (embed_dim=128) │
        │ + Residual Add & LayerNorm        │
        │ + FeedForward Network (dim=256)   │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │ Mask-Aware Mean Pooling           │
        │ h_fused = (∑ mᵢ·Zᵢ) / (∑ mᵢ)      │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │ 4-Way Classifier Head             │
        │ Linear(128, 64) -> ReLU -> (4)    │
        └─────────────────┬─────────────────┘
                          │
         ┌────────────────┼────────────────┬────────────────┐
         ▼                ▼                ▼                ▼
     BRCA (Breast)   HNSC (Head/Neck) LGG (Glioma)    THCA (Thyroid)
```

### Mathematical Formulation

1. **Modality Encoders:**
   $$\mathbf{z}_i = \text{MLP}_i(\mathbf{x}_i) = \mathbf{W}_{i,2} \cdot \text{ReLU}(\text{BN}(\mathbf{W}_{i,1} \mathbf{x}_i)) \in \mathbb{R}^{D}$$
2. **Missing Modality Imputation:**
   Given observation mask $m_i \in \{0, 1\}$:
   $$\tilde{\mathbf{z}}_i = m_i \mathbf{z}_i + (1 - m_i) \mathbf{e}_{\text{default}}^{(i)}$$
3. **Modality Type Injection:**
   $$\mathbf{Z}_0 = [\tilde{\mathbf{z}}_1, \dots, \tilde{\mathbf{z}}_M]^\top + \mathbf{E}_{\text{type}} \in \mathbb{R}^{M \times D}$$
4. **Multi-Head Cross-Attention:**
   $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}, \quad \mathbf{Z}_{\text{attn}} = \text{LayerNorm}(\mathbf{Z}_0 + \mathbf{A})$$
5. **Mask-Aware Pooling:**
   $$\mathbf{h}_{\text{fused}} = \frac{\sum_{i=1}^M m_i \mathbf{Z}_{\text{final}, i}}{\sum_{i=1}^M m_i + \epsilon} \in \mathbb{R}^D$$

---

## Clinical Web Simulator & Telemetry

Inside [`Code/`](file:///d:/PLACE/BCA/MajorProject/Onic/Code) is an interactive research client and FastAPI backend:

- **Live Modality Toggling:** Dynamically simulate 15 missingness scenarios in real time with immediate inference output.
- **Patient Case Selector:** Pre-loaded representative TCGA test patients spanning all 4 cancer subtypes.
- **Biomarker Attributions:** Visual breakdown of how much each observed modality contributed to the diagnostic prediction.
- **Dual-Engine Architecture:** Seamlessly connects to PyTorch via [`Code/api_server.py`](file:///d:/PLACE/BCA/MajorProject/Onic/Code/api_server.py) with an instant offline heuristic fallback.

```bash
# Launch the live API server
python Code/api_server.py

# In another terminal or browser, open Code/index.html
```

---

## Empirical Benchmarks (15 Combinatorial States)

Evaluated across **2,263 patient samples** (TCGA cohort) over 5 random seeds using stratified, patient-leak-free splits:

| Condition ID | Description | Present Modalities | Missing Modalities | S1 (Complete Fusion) | S2 (Modality Dropout) | Gain / Degradation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C0** | **Complete Multi-Omic** | **mRNA, miRNA, CNV, Methy** | *None* | **99.56%** | **99.45%** | Robust ceiling |
| **C1** | Missing mRNA | miRNA, CNV, Methy | mRNA | 98.92% | **99.45%** | +0.53% |
| **C2** | Missing miRNA | mRNA, CNV, Methy | miRNA | 99.12% | **99.45%** | +0.33% |
| **C3** | Missing CNV | mRNA, miRNA, Methy | CNV | 99.24% | **99.45%** | +0.21% |
| **C4** | Missing Methy | mRNA, miRNA, CNV | Methy | 98.80% | **99.10%** | +0.30% |
| **C5** | Missing mRNA + miRNA | CNV, Methy | mRNA, miRNA | 94.12% | **96.84%** | **+2.72%** |
| **C6** | Missing mRNA + CNV | miRNA, Methy | mRNA, CNV | 95.30% | **97.10%** | **+1.80%** |
| **C7** | Missing mRNA + Methy | miRNA, CNV | mRNA, Methy | 88.40% | **92.65%** | **+4.25%** |
| **C8** | Missing miRNA + CNV | mRNA, Methy | miRNA, CNV | 98.70% | **99.15%** | +0.45% |
| **C9** | Missing miRNA + Methy | mRNA, CNV | miRNA, Methy | 98.20% | **98.80%** | +0.60% |
| **C10** | Missing CNV + Methy | mRNA, miRNA | CNV, Methy | 98.50% | **98.90%** | +0.40% |
| **C11** | Methy Only (75% Missing) | Methy | mRNA, miRNA, CNV | 91.20% | **93.37%** | **+2.17%** |
| **C12** | **CNV Only (Stress Test)** | **CNV** | **mRNA, miRNA, Methy** | 65.60% | **74.12%** | **+8.52%** |
| **C13** | miRNA Only (75% Missing) | miRNA | mRNA, CNV, Methy | 82.40% | **87.20%** | **+4.80%** |
| **C14** | mRNA Only (75% Missing) | mRNA | miRNA, CNV, Methy | 97.60% | **98.40%** | +0.80% |

> [!IMPORTANT]
> **Key Finding:** Under single-modality stress tests where 75% of diagnostic information is missing (C12: CNV Only), our Modality-Dropout trained network achieves a **+8.52% boost** in Macro-F1 over standard fusion, preventing catastrophic failure.

---

## Latent Space & Diagnostic Visualizations

<p align="center">
  <img src="project/reports/presentation_figures/A1_confusion_matrix_normalized.png" width="48%" alt="Normalized Confusion Matrix"/>
  <img src="project/reports/presentation_figures/A2_multiclass_roc.png" width="48%" alt="Multiclass ROC Curves"/>
</p>

<p align="center">
  <img src="project/reports/presentation_figures/C1_fused_embeddings_tsne.png" width="48%" alt="t-SNE Fused Embeddings"/>
  <img src="project/reports/presentation_figures/B1_missing_condition_performance.png" width="48%" alt="Missing Condition Degradation Curve"/>
</p>

- **High Class Separability:** t-SNE analysis of the 128-dimensional metric space reveals distinct cluster formation for all 4 tumor types with minimal inter-class confusion.
- **Near-Perfect ROC Performance:** Macro-AUC reaches **0.9996**, showing exceptional sensitivity and specificity across BRCA (100%), LGG (100%), THCA (100%), and HNSC (97.4%).

---

## Repository Structure

```
Onic/
├── assets/                                 # Visual brand & animated diagrams
│   ├── banner.svg                          # High-resolution project banner
│   └── model_workflow.svg                  # Animated neural pipeline SVG
├── Code/                                   # Interactive Web Dashboard
│   ├── index.html                          # Clinical simulator UI
│   ├── style.css                           # Modern responsive CSS design system
│   ├── script.js                           # Real-time inference & telemetry client
│   ├── api_server.py                       # FastAPI PyTorch inference server
│   └── PRODUCT.md                          # Clinical application blueprint
├── project/                                # Machine Learning Research Engine
│   ├── configs/                            # Data & model YAML configurations
│   ├── docs/                               # 12+ Comprehensive research protocols
│   ├── src/                                # Core PyTorch modules
│   │   ├── data/                           # Leak-free splitting & loaders
│   │   ├── models/                         # MultimodalCrossAttentionClassifier
│   │   ├── training/                       # Seeded trainers & custom losses
│   │   └── explainability/                 # Modality & feature attribution
│   ├── scripts/                            # Reproducibility & audit runners
│   ├── reports/                            # Publication drafts, metrics, figures
│   │   └── presentation_figures/           # High-resolution presentation plots
│   └── requirements.txt                    # Python environment specifications
└── .gitignore                              # Git tracking exclusions
```

---

## Quickstart & Reproduction

### 1. Clone the Repository

```bash
git clone https://github.com/ubstorm128/Onic.git
cd Onic
```

### 2. Environment Setup

```bash
# Create and activate a Python virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r project/requirements.txt
```

### 3. Run Pipeline Audits & Baselines

```bash
# Verify dataset integrity and sample alignment (26/26 audit checks)
python project/scripts/audit_dataset.py

# Run baseline comparisons (MLPs, Early Fusion, LightGBM)
python project/scripts/run_baselines.py

# Run multimodal cross-attention fusion under missing modality dropout
python project/scripts/run_fusion.py
```

### 4. Launch the Interactive Web Application

```bash
# Start the FastAPI inference service
python Code/api_server.py

# Open Code/index.html in any modern web browser
```

---

## Academic Credits & Team

Developed as a Major Project for the **Bachelor of Computer Applications (BCA)** degree:

- **Department:** Faculty of Computer Technology
- **Institution:** [Assam down town University, Guwahati](https://adtu.in)
- **Academic Supervisor:** Mr. Wanraplang Nongbri

### Research & Development Team:
- **Aniket Saha** — *Data Engineering & Deep Learning Architecture*
- **Udit Bora** — *Lead Developer & Interface Engineering*
- **Kritartha Barman** — *Evaluation Protocols & Leak-Free Pipeline*

---

<p align="center">
  <sub>Built for biomedical machine learning and clinical diagnostic resilience.</sub>
</p>
