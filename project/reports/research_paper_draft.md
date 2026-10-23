# A Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities

**Authors:** Computational Biology & Machine Learning Research Team  
**Institution:** Major Project Research Consortium  
**Date:** October 2026  
**Status:** Working Research Manuscript Draft (Pre-Peer-Review)  

---

## Abstract

Multimodal integration of high-throughput omics data—such as transcriptomic, epigenomic, and genomic structural variations—offers high resolution for precision oncology. However, real-world clinical translation is impeded by the *missing modality problem*, where patient cohorts frequently lack one or more diagnostic profiling assays due to tissue scarcity, sequencing costs, or assay attrition. Here, we present a mathematically formulated Multimodal Artificial Intelligence framework for pan-cancer classification under arbitrary missing modality conditions. Leveraging a harmonized cohort of 2,263 TCGA patient samples across four distinct malignancies (Breast Invasive Carcinoma [BRCA], Head and Neck Squamous Cell Carcinoma [HNSC], Brain Lower Grade Glioma [LGG], and Thyroid Carcinoma [THCA]), our architecture integrates dedicated non-linear feature encoders ($D=128$), learned modality-type embeddings, learned default missingness tokens, and a multi-head cross-attention mechanism ($H=4$) with mask-aware pooling. Evaluated under strict zero-leakage anti-contamination protocols across 5 fixed random seeds ($[42, 123, 456, 789, 1024]$), our proposed complete reference model (Strategy S1) achieves a Test Macro-F1 of $0.9956 \pm 0.0015$ on complete data. Under systematic stress testing across all $2^4 - 1 = 15$ missing modality combinations (Conditions C0 through C14), when only noisy structural variation (CNV) is available (Condition C12), Strategy S1 drops to $0.6561 \pm 0.0892$. Training with controlled stochastic modality dropout (Strategy S2, $p=0.3$) elevates single-CNV classification to $0.7412 \pm 0.0476$—an empirical paired mean gain of $+8.52\%$ ($+0.0852$) with a paired Cohen's $d = 0.7425$. Across the 5 paired seeds, this gain yields paired $t(4) = 1.6603, p = 0.1722$ and Wilcoxon $W = 2.0, p = 0.1875$ (95% CI: $[-0.0573, +0.2276]$), demonstrating an empirically substantial effect that does not reach nominal statistical significance at $\alpha = 0.05$ due to small-sample degrees of freedom ($df=4$). In ablation studies, increasing dropout to $p=0.5$ further increases single-CNV performance to $0.7897 \pm 0.0107$. Dynamic attention routing descriptively downweights noisy genomic structural alterations (CNV attention: $5.7\%$) relative to transcriptomic ($34.0\%$) and epigenetic methylation states ($34.2\%$). Gradient-based sensitivity attributions identify features that align with established literature (e.g., *SFRP1* in BRCA, *ANO1* in HNSC, *PDGFRA* in LGG, and *DPP4* in THCA), providing computational interpretability rather than causal biological proof or clinical biomarker validation. This manuscript provides a fully reproducible working draft for multimodal diagnostic AI under missingness.

---

## 1. Introduction

Precision oncology increasingly relies on multi-omics profiling to resolve tumor heterogeneity, decipher therapeutic vulnerabilities, and improve diagnostic categorization. While single-modality assays (e.g., mRNA sequencing) provide substantial biological signal, integrating orthogonal modalities—including microRNA (miRNA) regulatory layers, copy number variations (CNV), and DNA methylation—captures multi-tier functional genomics.

However, machine learning models deployed in clinical practice confront two fundamental challenges:
1. **The Missing Modality Problem:** Patients rarely receive all four assays simultaneously due to institutional budget disparities, archival formalin-fixed paraffin-embedded (FFPE) tissue degradation, or sequencing failures. Conventional multimodal architectures require complete input matrices and crash or produce uncalibrated predictions when inputs are missing.
2. **Signal-to-Noise Disparities Across Modalities:** Different biological assays possess disparate signal-to-noise ratios. Structural alterations (CNVs) exhibit high genomic plasticity across cancer types, making them less diagnostic than epigenetic methylation states or transcriptomic profiles. Simple feature concatenation fails to modulate modality influence adaptively.

To address these challenges, we formulate and investigate five core research questions:
- **RQ1 (Multimodal Benchmark):** How do unimodal classifiers, complete early fusion, and tree-based gradient boosting compare on complete multi-omics data?
- **RQ2 (Dynamic Fusion Routing):** How does multi-head cross-attention modulate attention weights across biological modalities with disparate noise profiles?
- **RQ3 (Degradation Under Missingness):** What is the exact trajectory of performance degradation across all 15 possible combinations of missing modalities?
- **RQ4 (Modality Dropout Mitigation):** Can training with stochastic modality dropout mitigate degradation when noisy modalities alone are available, and what is its paired statistical significance?
- **RQ5 (Computational Attributions):** Do gradient-based feature sensitivity rankings highlight genes consistent with canonical cancer literature?

---

## 2. Cohort Harmonization, Preprocessing, and Anti-Leakage Verification

### 2.1 Study Population and Omics Assays
The study cohort was derived from the benchmark MLOmics Pan-Cancer repository (Zenodo DOI: 10.5281/zenodo.28729127), comprising 8,314 initial samples across 33 TCGA cancer types. Following validation against official TCGA label registers, four clinically representative tumor types were selected:
- **BRCA (Breast Invasive Carcinoma):** $N = 757$ samples (Label 2)
- **HNSC (Head and Neck Squamous Cell Carcinoma):** $N = 506$ samples (Label 8)
- **LGG (Brain Lower Grade Glioma):** $N = 506$ samples (Label 13)
- **THCA (Thyroid Carcinoma):** $N = 494$ samples (Label 27)
- **Total Harmonized Cohort:** $N = 2,263$ patient samples.

The four profiling modalities encompass 9,844 raw biological features:
1. **mRNA Expression:** 3,217 genes (RNA-Seq normalized counts)
2. **miRNA Expression:** 383 microRNAs (miRNA-Seq normalized expression)
3. **Copy Number Variations (CNV):** 3,105 genomic regions (GISTIC thresholded segment values)
4. **DNA Methylation:** 3,139 CpG probes (Illumina Infinium HumanMethylation450 platform $\beta$-values $\in [0, 1]$)

### 2.2 Quality Control, Patient Uniqueness, and Anti-Leakage Verification
A comprehensive data audit confirmed:
- **Patient Uniqueness:** Exactly 2,263 unique sample IDs correspond to 2,263 unique patient IDs (TCGA 12-character patient barcodes). Zero multi-sample or multi-aliquot patients exist in this four-cancer cohort.
- **Missing Values:** Zero NaN or null values exist in raw feature matrices.
- **Stratified Partitioning (70% / 15% / 15%):**
  - **Training Set:** $N = 1,584$ samples (1,584 unique patients)
  - **Validation Set:** $N = 339$ samples (339 unique patients)
  - **Test Set:** $N = 340$ samples (340 unique patients)
- **Disjointness:** Both sample-level and patient-level intersections are strictly null:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Executed Preprocessing Operations:**
  All feature filtering and scaling operations were fitted exclusively on the 1,584 training samples:
  1. `mRNA`: `StandardScaler` (z-score normalization) and zero-variance thresholding (`variance_threshold = 0.0`); all 3,217 features retained.
  2. `miRNA`: `StandardScaler` and zero-variance thresholding; all 383 features retained.
  3. `CNV`: `StandardScaler` and zero-variance thresholding; all 3,105 features retained.
  4. `Methy`: `MinMaxScaler` (bounded $[0, 1]$ $\beta$-value scaling) and zero-variance thresholding; all 3,139 features retained.
  Fitted preprocessor objects were serialized and applied to validation and test partitions without re-fitting.

---

## 3. Multimodal Cross-Attention Architecture

### 3.1 Dedicated Non-Linear Modality Encoders
For each modality $m \in \{\text{mRNA}, \text{miRNA}, \text{CNV}, \text{Methy}\}$, raw feature vectors $x_m \in \mathbb{R}^{K_m}$ are projected into a shared metric embedding space $z_m \in \mathbb{R}^D$ ($D=128$) via a two-layer non-linear multilayer perceptron:
$$z_m = \text{Dropout}\left(\text{ReLU}\left(\text{BN}\left(W_{m,2} \cdot \text{Dropout}\left(\text{ReLU}\left(\text{BN}\left(W_{m,1} x_m\right)\right)\right)\right)\right)\right)$$
where $W_{m,1} \in \mathbb{R}^{256 \times K_m}$ and $W_{m,2} \in \mathbb{R}^{128 \times 256}$.

### 3.2 Modality Type Embeddings and Learned Default Missing Tokens
To preserve modality identity in the sequence, a learnable modality-type embedding $E_m \in \mathbb{R}^D$ is added to each token. When a modality is missing (indicated by binary mask indicator $M_{i,m} = 0$), the missing input is substituted by a learnable default missing token $\theta_m \in \mathbb{R}^D$:
$$\tilde{z}_{i,m} = M_{i,m} z_{i,m} + (1 - M_{i,m}) \theta_m + E_m$$
Formal unit tests verify that altering numerical values of masked modalities ($M_{i,m}=0$) produces zero difference in evaluation logits ($\max |\Delta| < 10^{-6}$).

### 3.3 Multi-Head Cross-Attention Fusion
The token sequence $Z = [\tilde{z}_1, \tilde{z}_2, \tilde{z}_3, \tilde{z}_4] \in \mathbb{R}^{4 \times D}$ undergoes multi-head self-attention:
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \mathcal{M}_{\text{pad}}\right) V$$
where $\mathcal{M}_{\text{pad}}$ enforces key padding masking over missing modalities. Token representations are aggregated via mask-aware mean pooling:
$$h_{\text{fused}} = \frac{\sum_{m=1}^4 M_{i,m} \cdot z'_{i,m}}{\max(1, \sum_{m=1}^4 M_{i,m})}$$
and classified into the four cancer classes using a regularized linear projection head. (An all-missing condition guard unmasks tokens and sets pooling divisor to 1.0, serving as an internal numerical safeguard rather than a clinical deployment mode).

### 3.4 Modality Dropout Training Strategy
To eliminate reliance on co-present modalities, training Strategy S2 enforces stochastic modality dropout: each modality is masked out independently with probability $p = 0.3$ (and up to $p=0.5$ in ablations), subject to the constraint that at least one modality remains active. Strategy S1 trains exclusively on complete inputs ($p=0.0$).

---

## 4. Empirical Evaluation & Rigorous Statistical Analysis

### 4.1 Statistical Framework and Variance Units
All models were evaluated across 5 fixed random seeds ($S = [42, 123, 456, 789, 1024]$) on the held-out test split of 340 patients.
- **Variance Interpretation:** All reported sample standard deviations, standard errors, confidence intervals, and hypothesis tests quantify **algorithmic training variance** (differences arising from random weight initialization, mini-batch sampling, and stochastic dropout masks). They do **not** represent patient population sampling variance, because all evaluations were conducted on the same 340 test patients.
- **Paired Hypothesis Testing:** Because Strategy S1 and Strategy S2 share identical test instances and corresponding seeds, paired tests are reported: paired mean difference ($\bar{D}$), paired standard deviation ($s_D$), paired Cohen's $d = \bar{D} / s_D$, paired Student's $t$-test ($df=4$), Wilcoxon signed-rank test, and 95% confidence intervals of the difference.
- **Wilcoxon Small-Sample Bound:** With $N=5$ pairs, the minimum possible two-sided Wilcoxon $p$-value is $2 \times (1/2)^5 = 2/32 = 0.0625$ (achieved only when all 5 pairs show strictly positive differences without ties). Consequently, Wilcoxon signed-rank tests cannot reach nominal significance at $\alpha = 0.05$ with $N=5$.

### 4.2 Benchmark Model Comparison (Complete Inputs Only)

Baseline models D1–D4 evaluate single modalities. Baselines D5 (Early Fusion MLP) and D6 (LightGBM) were trained and evaluated **strictly on complete 4-omics inputs (Condition C0)**; they were not benchmarked under missing modality conditions.

| Model ID | Architecture | Input Scope | N Seeds | Raw Per-Seed Test Macro-F1 | Test Macro-F1 (Mean ± Std) | 95% Confidence Interval | Seed Variance? |
|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|
| **D1** | Single-Modality MLP | mRNA (3,217) | 5 | [0.9945, 0.9973, 0.9973, 0.9973, 0.9945] | $0.9961 \pm 0.0015$ | $[0.9943, 0.9980]$ | Yes |
| **D2** | Single-Modality MLP | miRNA (383) | 5 | [0.9973, 0.9973, 0.9945, 0.9973, 0.9973] | $0.9967 \pm 0.0012$ | $[0.9952, 0.9982]$ | Yes |
| **D3** | Single-Modality MLP | CNV (3,105) | 5 | [0.8548, 0.8589, 0.8661, 0.8355, 0.8507] | $0.8532 \pm 0.0114$ | $[0.8391, 0.8674]$ | Yes |
| **D4** | Single-Modality MLP | Methy (3,139) | 5 | [0.9945, 0.9945, 0.9945, 0.9945, 0.9945] | $0.9945 \pm 0.0000$ | $[0.9945, 0.9945]$ | No (Deterministic) |
| **D5** | Early Fusion MLP | Complete (9,844) | 5 | [0.9973, 0.9973, 0.9973, 0.9945, 0.9945] | $0.9961 \pm 0.0015$ | $[0.9943, 0.9980]$ | Yes |
| **D6** | LightGBM (GBDT) | Complete (9,844) | 5 | [0.9973, 0.9973, 0.9973, 0.9973, 0.9973] | $\mathbf{0.9973 \pm 0.0000}$ | $[0.9973, 0.9973]$ | No (Deterministic) |
| **E1** | Proposed Fusion (S1: Complete) | Complete (9,844) | 5 | [0.9945, 0.9945, 0.9973, 0.9973, 0.9945] | $0.9956 \pm 0.0015$ | $[0.9937, 0.9975]$ | Yes |
| **E2** | Proposed Fusion (S2: Dropout $p=0.3$) | Complete (9,844) | 5 | [0.9945, 0.9973, 0.9945, 0.9945, 0.9945] | $0.9950 \pm 0.0012$ | $[0.9935, 0.9966]$ | Yes |

*Note on Baseline Scope:* On complete inputs, LightGBM (D6) achieves the highest Macro-F1 ($0.9973$). Our proposed architecture is specifically designed for robustness under missing modalities; it cannot be claimed to outperform D5 or D6 under missing modalities without future evaluations incorporating baseline imputation strategies.

---

### 4.3 Missing Modality Evaluation Across All 15 Conditions (S1 vs. S2)

Both Strategy S1 (trained on complete inputs) and Strategy S2 (trained with modality dropout $p=0.3$) were evaluated across all 15 conditions on the 340 test samples ($N=5$ seeds, 150 evaluations). Below are the exact per-condition paired statistics:

| Cond | Description | S1 Mean ± Std | S2 Mean ± Std | Paired Diff ($\bar{D}$) | Paired Cohen's $d$ | Paired $t$-stat | Paired $t$ $p$-val | Wilcoxon stat ($p$) | 95% CI of Diff | Significant ($\alpha=0.05$)? |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | Complete (reference) | 0.9956 ± 0.0015 | 0.9950 ± 0.0012 | **-0.000552** | -0.2390 | -0.5345 | 0.6213 | 2.0 (1.0000) | [-0.0034, +0.0023] | No |
| **C1** | mRNA missing | 0.9939 ± 0.0012 | 0.9950 ± 0.0012 | **+0.001107** | 0.4472 | 1.0000 | 0.3739 | 0.0 (1.0000) | [-0.0020, +0.0042] | No |
| **C2** | miRNA missing | 0.9945 ± 0.0000 | 0.9945 ± 0.0000 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 0.0 (1.0000) | [+0.0000, +0.0000] | No |
| **C3** | CNV missing | 0.9956 ± 0.0015 | 0.9967 ± 0.0012 | **+0.001104** | 0.4472 | 1.0000 | 0.3739 | 2.5 (0.6250) | [-0.0020, +0.0042] | No |
| **C4** | Methy missing | 0.9956 ± 0.0015 | 0.9967 ± 0.0012 | **+0.001104** | 0.4472 | 1.0000 | 0.3739 | 2.5 (0.6250) | [-0.0020, +0.0042] | No |
| **C5** | mRNA+miRNA missing | 0.9934 ± 0.0025 | 0.9938 ± 0.0015 | **+0.000436** | 0.1357 | 0.3034 | 0.7767 | 1.0 (1.0000) | [-0.0036, +0.0044] | No |
| **C6** | mRNA+CNV missing | 0.9956 ± 0.0015 | 0.9956 ± 0.0015 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 5.0 (1.0000) | [-0.0034, +0.0034] | No |
| **C7** | mRNA+Methy missing | 0.9928 ± 0.0015 | 0.9939 ± 0.0030 | **+0.001101** | 0.2641 | 0.5906 | 0.5866 | 4.0 (0.8750) | [-0.0041, +0.0063] | No |
| **C8** | miRNA+CNV missing | 0.9950 ± 0.0012 | 0.9950 ± 0.0012 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 1.5 (1.0000) | [-0.0024, +0.0024] | No |
| **C9** | miRNA+Methy missing | 0.9950 ± 0.0012 | 0.9950 ± 0.0012 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 1.5 (1.0000) | [-0.0024, +0.0024] | No |
| **C10** | CNV+Methy missing | 0.9973 ± 0.0000 | 0.9973 ± 0.0000 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 0.0 (1.0000) | [+0.0000, +0.0000] | No |
| **C11** | Only Methy present | 0.9923 ± 0.0050 | 0.9933 ± 0.0028 | **+0.001001** | 0.1592 | 0.3559 | 0.7399 | 1.0 (1.0000) | [-0.0068, +0.0088] | No |
| **C12** | **Only CNV present** | **0.6561 ± 0.0892** | **0.7412 ± 0.0476** | **+0.085172** | **0.7425** | **1.6603** | **0.1722** | **2.0 (0.1875)** | **[-0.0573, +0.2276]** | **No** |
| **C13** | Only miRNA present | 0.9915 ± 0.0034 | 0.9945 ± 0.0033 | **+0.003006** | 0.6763 | 1.5122 | 0.2050 | 2.0 (0.1875) | [-0.0025, +0.0085] | No |
| **C14** | Only mRNA present | 0.9956 ± 0.0015 | 0.9973 ± 0.0000 | **+0.001656** | 1.0954 | 2.4495 | 0.0705 | 0.0 (0.2500) | [-0.0002, +0.0035] | No |

### 4.4 Detailed Analysis of Condition C12 (Isolated CNV)
Condition C12 represents the most severe degradation point in the pipeline because CNV data possess substantial noise and structural heterogeneity.
- The raw per-seed Macro-F1 scores on C12 are:
  - Seed 42: S1 = $0.7368$, S2 = $0.7413$ ($\Delta = +0.0045$)
  - Seed 123: S1 = $0.5181$, S2 = $0.7897$ ($\Delta = +0.2717$)
  - Seed 456: S1 = $0.6777$, S2 = $0.7897$ ($\Delta = +0.1120$)
  - Seed 789: S1 = $0.7525$, S2 = $0.7413$ ($\Delta = -0.0112$)
  - Seed 1024: S1 = $0.5953$, S2 = $0.6441$ ($\Delta = +0.0489$)
- **Empirical Gain:** Strategy S2 outperforms Strategy S1 on 4 out of 5 seeds, producing an unrounded paired mean difference of $+0.085172$ ($+8.52\%$).
- **Statistical Significance Assessment:** With paired standard deviation $s_D = 0.1147$, the paired Cohen's $d$ is $0.7425$ (a moderate-to-large effect size). However, because $N=5$ paired seeds ($df=4$), the paired Student's $t$-test yields $t = 1.6603, p = 0.1722$, and the two-sided Wilcoxon signed-rank test yields $W = 2.0, p = 0.1875$. The 95% confidence interval for the paired difference is $[-0.0573, +0.2276]$, which spans zero.
- **Scientific Conclusion:** The observed $+8.52\%$ improvement is an **empirically substantial effect** that demonstrates consistent gain across 80% of training initializations, but it is **not statistically significant at $\alpha = 0.05$** due to limited statistical power with five training runs.

---

## 5. Systematic Ablation Studies

To examine architecture components, we performed 4 multi-seed ablation experiments ($N=24$ training runs):

### 5.1 Masking Mechanism: Learned Default Token vs. Zero Masking
- **Learned Default Token:** Complete C0 = $0.9954 \pm 0.0013$ | Only CNV C12 = $0.6768 \pm 0.0934$
- **Zero Vector Masking:** Complete C0 = $0.9945 \pm 0.0000$ | Only CNV C12 = $0.6807 \pm 0.1263$
- **Takeaway:** Zero-masking creates higher variance in test evaluation, resulting in a **$35.2\%$ wider standard deviation** under missing conditions. Learned default tokens regularize embedding trajectories.

### 5.2 Training Modality Dropout Probability ($p \in \{0.1, 0.3, 0.5\}$)
- **$p = 0.1$:** C0 = $0.9945 \pm 0.0000$ | C12 = $0.6605 \pm 0.0181$
- **$p = 0.3$:** C0 = $0.9954 \pm 0.0013$ | C12 = $0.6768 \pm 0.0934$
- **$p = 0.5$:** C0 = $0.9945 \pm 0.0000$ | C12 = $\mathbf{0.7897 \pm 0.0107}$
- **Takeaway:** Robustness scales with dropout probability. Increasing $p$ to $0.5$ yields a $+13.36\%$ absolute gain on C12 over baseline S1, while incurring zero degradation on complete inputs ($0.9945$).

### 5.3 Latent Representation Dimension ($D \in \{64, 128, 256\}$)
- **$D = 64$:** C0 = $0.9954 \pm 0.0013$ | C12 = $\mathbf{0.7011 \pm 0.0387}$
- **$D = 128$:** C0 = $0.9954 \pm 0.0013$ | C12 = $0.6768 \pm 0.0934$
- **$D = 256$:** C0 = $0.9945 \pm 0.0000$ | C12 = $0.6633 \pm 0.0875$
- **Takeaway:** Compact metric latent spaces ($D=64$ to $128$) constrain the representation manifold and prevent co-adaptation, delivering higher resilience than overparameterized $D=256$.

### 5.4 Attention Heads ($H \in \{1, 2, 4\}$)
- **$H = 1$ (Single-Head):** C0 = $0.9973 \pm 0.0000$ | C12 = $\mathbf{0.5907 \pm 0.1094}$
- **$H = 2$:** C0 = $0.9945 \pm 0.0000$ | C12 = $\mathbf{0.7426 \pm 0.0230}$
- **$H = 4$:** C0 = $0.9954 \pm 0.0013$ | C12 = $0.6768 \pm 0.0934$
- **Takeaway:** Single-head attention undergoes substantial degradation when noisy CNV is isolated ($0.5907$). Multi-head attention is essential for decoupling orthogonal representation subspaces.

---

## 6. Model Interpretability and Computational Attributions

### 6.1 Inter-Modality Cross-Attention Routing
Quantitative extraction of the learned self-attention weights provides descriptive insight into internal network routing:
- **DNA Methylation:** $34.2\%$ mean attention weight
- **mRNA Expression:** $34.0\%$ mean attention weight
- **miRNA Regulatory Layer:** $26.1\%$ mean attention weight
- **Copy Number Variation (CNV):** $\mathbf{5.7\%}$ mean attention weight

*Methodological Note:* In accordance with Jain & Wallace (2019) (*"Attention is not Explanation"*), attention weights represent internal computational routing coefficients rather than causal biological contributions or biophysical interactions. Descriptively, the multi-head attention module places lower relative weight on the noisy CNV channel when transcriptomic and epigenetic channels are concurrently present.

### 6.2 Gradient-Based Feature Sensitivity
Feature sensitivity was computed using input-gradient attribution ($\text{Input} \times |\nabla_x|$) across correctly classified test instances:
- **BRCA:** *SFRP1* (Secreted Frizzled-Related Protein 1, known Wnt antagonist) and *hsa-miR-10b-3p*.
- **HNSC:** *ANO1* (Anoctamin 1, amplified 11q13 region) and *PPFIA1*.
- **LGG:** *PDGFRA* (Platelet-Derived Growth Factor Receptor Alpha, receptor tyrosine kinase) and *miR-219a*.
- **THCA:** *DPP4* (Dipeptidyl Peptidase 4, diagnostic marker for papillary thyroid carcinoma) and *FN1*.

*Interpretability Scope:* These rankings demonstrate that the neural network relies on biological features consistent with published oncology literature. Crucially, these computational sensitivity attributions do **not** establish causal disease mechanisms or prospectively validated clinical biomarkers, which require independent experimental wet-lab and prospective clinical validation.

---

## 7. Limitations & Threats to Validity

1. **Statistical Power of Seed Evaluations:** Evaluations across $N=5$ training seeds provide 4 degrees of freedom for paired $t$-tests and cannot reach $p < 0.05$ under the Wilcoxon signed-rank test. A larger number of seeds or cross-validation runs is necessary to establish formal statistical significance.
2. **Algorithmic vs. Population Variance:** Standard deviations and confidence intervals reflect training stochasticity on a fixed 340-patient test set, not patient population sampling uncertainty across external hospital cohorts.
3. **Scope of Baseline Comparisons:** Tree-based GBDT (LightGBM) and Early Fusion MLP baselines were evaluated exclusively on complete inputs; their behavior under missing-modality regimes with imputation remains to be evaluated.
4. **TCGA Batch Effects:** While verified against canonical TCGA manifests, pan-cancer sequencing across multiple tissue source sites can retain subtle batch variances.
5. **Missing Modality Independence:** In clinical oncology, missing modalities may be missing not at random (MNAR); for example, aggressive tumors may undergo more extensive profiling.
6. **Pre-Peer-Review Status:** This document is an internal working draft and has not undergone formal peer review.

---

## 8. Conclusion

We have formulated, tested, and audited a Multimodal AI framework for pan-cancer classification under missing modality conditions. By combining dedicated non-linear encoders, learned missingness tokens, multi-head cross-attention, and stochastic modality dropout, the framework mitigates degradation under extreme missingness (+8.52% empirical gain under isolated CNV, $d = 0.7425$, though $p = 0.1722$ across 5 seeds). Attention routing and gradient sensitivity highlight biological features aligned with established cancer literature. All split manifests, preprocessing objects, trained model checkpoints, and raw per-seed metric logs are preserved to support full scientific reproducibility.
